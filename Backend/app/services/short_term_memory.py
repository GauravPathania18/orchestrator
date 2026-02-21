"""
Short-term memory with automatic time-based session management.

Features:
- Automatic session creation and management based on time
- Sessions expire after 60min inactivity -> get summarized and stored in vector DB
- Semantic summarization combines similar conversations
- No manual session_id needed - automatically managed
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
import uuid
import asyncio
from app.services import vector_client


class TimeBasedSessionManager:
    """
    Manages chat sessions automatically based on time.
    
    - Creates new sessions automatically
    - Expires sessions after 60min inactivity
    - Summarizes expired sessions semantically
    - Stores summaries in vector DB for long-term memory
    """
    
    def __init__(
        self,
        session_timeout_minutes: int = 60,
        max_messages: int = 20,
        max_context_chars: int = 4000
    ):
        self._sessions: Dict[str, List[Dict]] = {}  # session_id -> messages
        self._session_metadata: Dict[str, Dict] = {}  # session_id -> {created_at, last_accessed}
        self._lock = threading.Lock()
        self._timeout = timedelta(minutes=session_timeout_minutes)
        self.max_messages = max_messages
        self.max_context_chars = max_context_chars
        
        # Track current active session
        self._current_session_id: Optional[str] = None
        self._last_activity: Optional[datetime] = None

    def _generate_session_id(self) -> str:
        """Generate a new unique session ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"session_{timestamp}_{unique_id}"

    def _estimate_chars(self, message: str) -> int:
        """Estimate character count (simple proxy for tokens)."""
        return len(message)

    def _trim_context(self, session_id: str):
        """Trim session to fit within limits. Removes oldest first."""
        if session_id not in self._sessions:
            return
        
        messages = self._sessions[session_id]
        original_count = len(messages)
        
        # Trim by message count
        while len(messages) > self.max_messages:
            messages.pop(0)
        
        # Add summary indicator if trimming occurred
        if len(messages) < original_count:
            messages.insert(0, {
                "role": "system",
                "message": f"[Context trimmed: {original_count - len(messages)} older messages removed]",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"type": "context_summary"}
            })
        
        # Trim by character count
        total_chars = sum(self._estimate_chars(m.get("message", "")) for m in messages)
        while total_chars > self.max_context_chars and len(messages) > 1:
            removed = messages.pop(0)
            if removed.get("metadata", {}).get("type") == "context_summary":
                continue
            total_chars -= self._estimate_chars(removed.get("message", ""))

    async def _summarize_session(self, session_id: str) -> str:
        """
        Create semantic summary of session conversations.
        Groups similar messages and creates a condensed summary.
        """
        if session_id not in self._sessions:
            return ""
        
        messages = self._sessions[session_id]
        if not messages:
            return ""
        
        # Extract user and assistant messages (skip system)
        conversation_pairs = []
        current_pair = {"user": None, "assistant": None}
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("message", "")
            
            if role == "user":
                if current_pair["user"]:
                    conversation_pairs.append(current_pair.copy())
                current_pair = {"user": content, "assistant": None}
            elif role == "assistant":
                current_pair["assistant"] = content
        
        if current_pair["user"]:
            conversation_pairs.append(current_pair)
        
        if not conversation_pairs:
            return ""
        
        # Create semantic summary by grouping similar topics
        summary_parts = []
        current_topic = []
        
        for pair in conversation_pairs:
            user_msg = pair["user"] or ""
            if current_topic and self._are_messages_related(current_topic[-1]["user"], user_msg):
                current_topic.append(pair)
            else:
                if current_topic:
                    summary_parts.append(self._summarize_topic_group(current_topic))
                current_topic = [pair]
        
        if current_topic:
            summary_parts.append(self._summarize_topic_group(current_topic))
        
        # Combine all summaries
        full_summary = f"Session {session_id} Summary:\n" + "\n".join(summary_parts)
        
        # Store in vector DB
        await self._store_summary_in_vector_db(session_id, full_summary, messages)
        
        return full_summary

    def _are_messages_related(self, msg1: str, msg2: str) -> bool:
        """Simple check if two messages are semantically related."""
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "must", "shall", "can",
                     "need", "dare", "ought", "used", "to", "of", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through",
                     "during", "before", "after", "above", "below", "between",
                     "under", "and", "but", "or", "yet", "so", "if", "because",
                     "although", "though", "while", "where", "when", "that",
                     "which", "who", "whom", "whose", "what", "this", "these",
                     "those", "i", "me", "my", "mine", "myself", "you", "your",
                     "yours", "yourself", "he", "him", "his", "himself", "she",
                     "her", "hers", "herself", "it", "its", "itself", "we", "us",
                     "our", "ours", "ourselves", "they", "them", "their", "theirs",
                     "themselves", "am", "being", "been"}
        
        keywords1 = words1 - stop_words
        keywords2 = words2 - stop_words
        
        if not keywords1 or not keywords2:
            return False
        
        overlap = keywords1 & keywords2
        similarity = len(overlap) / max(len(keywords1), len(keywords2))
        
        return similarity > 0.3

    def _summarize_topic_group(self, pairs: List[Dict]) -> str:
        """Summarize a group of related conversation pairs."""
        if not pairs:
            return ""
        
        themes = []
        for pair in pairs:
            user_msg = pair.get("user", "")
            words = user_msg.split()[:5]
            theme = " ".join(words)
            if len(theme) > 50:
                theme = theme[:50] + "..."
            themes.append(theme)
        
        unique_themes = list(set(themes))
        
        if len(pairs) == 1:
            return f"- Discussed: {unique_themes[0]}"
        else:
            return f"- Discussed {len(pairs)} related topics including: {', '.join(unique_themes[:3])}"

    async def _store_summary_in_vector_db(self, session_id: str, summary: str, messages: List[Dict]):
        """Store session summary in vector DB for long-term retrieval."""
        try:
            metadata = {
                "type": "session_summary",
                "session_id": session_id,
                "message_count": len(messages),
                "timestamp": datetime.now().isoformat(),
                "summary_of": "conversation_session"
            }
            
            await vector_client.add_text(summary, metadata)
            print(f"[SessionManager] Stored summary for session {session_id} in vector DB")
        except Exception as e:
            print(f"[SessionManager] Error storing summary: {e}")

    def _is_session_expired(self, session_id: str) -> bool:
        """Check if a session has expired due to inactivity."""
        if session_id not in self._session_metadata:
            return True
        
        last_accessed = self._session_metadata[session_id].get("last_accessed")
        if not last_accessed:
            return True
        
        return datetime.now() - last_accessed > self._timeout

    async def get_or_create_session(self) -> str:
        """
        Get current active session or create new one.
        If current session expired, summarize it and create new.
        """
        with self._lock:
            if self._current_session_id and not self._is_session_expired(self._current_session_id):
                self._session_metadata[self._current_session_id]["last_accessed"] = datetime.now()
                self._last_activity = datetime.now()
                return self._current_session_id
            
            expired_session = None
            if self._current_session_id and self._is_session_expired(self._current_session_id):
                expired_session = self._current_session_id
                self._current_session_id = None
        
        if expired_session:
            print(f"[SessionManager] Session {expired_session} expired, summarizing...")
            await self._summarize_session(expired_session)
            with self._lock:
                self._sessions.pop(expired_session, None)
                self._session_metadata.pop(expired_session, None)
        
        with self._lock:
            new_session_id = self._generate_session_id()
            self._sessions[new_session_id] = []
            self._session_metadata[new_session_id] = {
                "created_at": datetime.now(),
                "last_accessed": datetime.now()
            }
            self._current_session_id = new_session_id
            self._last_activity = datetime.now()
            print(f"[SessionManager] Created new session: {new_session_id}")
            return new_session_id

    async def add_message(
        self,
        role: str,
        message: str,
        metadata: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Add message to session. Auto-manages session lifecycle.
        Returns the session_id used.
        """
        if session_id:
            active_session = session_id
            with self._lock:
                if session_id not in self._sessions:
                    self._sessions[session_id] = []
                    self._session_metadata[session_id] = {
                        "created_at": datetime.now(),
                        "last_accessed": datetime.now()
                    }
                self._session_metadata[session_id]["last_accessed"] = datetime.now()
        else:
            active_session = await self.get_or_create_session()
        
        with self._lock:
            self._sessions[active_session].append({
                "role": role,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            self._trim_context(active_session)
            self._session_metadata[active_session]["last_accessed"] = datetime.now()
            self._last_activity = datetime.now()
        
        return active_session

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Get messages for a specific session."""
        with self._lock:
            if session_id in self._session_metadata:
                self._session_metadata[session_id]["last_accessed"] = datetime.now()
            return self._sessions.get(session_id, []).copy()

    def get_current_session(self) -> Optional[str]:
        """Get current active session ID if not expired."""
        with self._lock:
            if self._current_session_id and not self._is_session_expired(self._current_session_id):
                return self._current_session_id
            return None

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session metadata."""
        with self._lock:
            if session_id not in self._session_metadata:
                return None
            
            meta = self._session_metadata[session_id]
            messages = self._sessions.get(session_id, [])
            
            created_at = meta.get("created_at")
            last_accessed = meta.get("last_accessed")
            
            return {
                "session_id": session_id,
                "created_at": created_at.isoformat() if created_at else None,
                "last_accessed": last_accessed.isoformat() if last_accessed else None,
                "message_count": len(messages),
                "is_expired": self._is_session_expired(session_id),
                "is_current": session_id == self._current_session_id
            }

    def get_all_sessions(self) -> List[Dict]:
        """Get info for all sessions."""
        with self._lock:
            sessions = []
            for sid in self._sessions.keys():
                info = self.get_session_info(sid)
                if info:
                    sessions.append(info)
            return sessions

    async def force_summarize_current(self) -> str:
        """Force summarize current session and start new one."""
        with self._lock:
            current = self._current_session_id
            self._current_session_id = None
        
        if current:
            await self._summarize_session(current)
            with self._lock:
                self._sessions.pop(current, None)
                self._session_metadata.pop(current, None)
        
        return await self.get_or_create_session()

    async def cleanup_all_expired(self):
        """Summarize and cleanup all expired sessions."""
        expired_sessions = []
        
        with self._lock:
            for session_id in list(self._sessions.keys()):
                if self._is_session_expired(session_id):
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            print(f"[SessionManager] Cleaning up expired session: {session_id}")
            await self._summarize_session(session_id)
            with self._lock:
                self._sessions.pop(session_id, None)
                self._session_metadata.pop(session_id, None)
                if self._current_session_id == session_id:
                    self._current_session_id = None

    # Backward compatibility aliases
    def get_context_window(self, session_id: str) -> List[Dict]:
        return self.get_session_messages(session_id)

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.get_session_messages(session_id)

    def clear_session(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._session_metadata.pop(session_id, None)
            if self._current_session_id == session_id:
                self._current_session_id = None

    def search_session(self, session_id: str, query: str) -> List[Dict]:
        if not session_id or not query:
            return []
        
        messages = self.get_session_messages(session_id)
        query_lower = query.lower()
        
        matches = []
        for msg in messages:
            message_text = msg.get("message", "").lower()
            if query_lower in message_text:
                matches.append(msg)
        
        return matches

    def get_context_stats(self, session_id: str) -> Dict[str, Any]:
        messages = self.get_session_messages(session_id)
        total_chars = sum(len(m.get("message", "")) for m in messages)
        
        msg_util = (len(messages) / self.max_messages) * 100 if self.max_messages else 0
        char_util = (total_chars / self.max_context_chars) * 100 if self.max_context_chars else 0
        
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "total_chars": total_chars,
            "max_messages": self.max_messages,
            "max_context_chars": self.max_context_chars,
            "utilization_percent": round(max(msg_util, char_util), 2),
            "is_full": len(messages) >= self.max_messages or total_chars >= self.max_context_chars
        }

    def set_context_limits(self, max_messages: Optional[int] = None, max_chars: Optional[int] = None):
        with self._lock:
            if max_messages is not None:
                self.max_messages = max_messages
            if max_chars is not None:
                self.max_context_chars = max_chars
            
            for session_id in self._sessions:
                self._trim_context(session_id)


# Global session manager instance
session_manager = TimeBasedSessionManager(
    session_timeout_minutes=60,
    max_messages=20,
    max_context_chars=4000
)

# Backward compatibility alias
short_term_memory = session_manager
