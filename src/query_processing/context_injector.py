"""
Context Injection Module

Manages conversation context and session state:
- Conversation history tracking
- Entity resolution across turns
- Anaphora resolution (pronouns, references)
- Context persistence
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    query: str
    entities: Dict[str, List]
    intent: Dict
    timestamp: str
    response_summary: Optional[str] = None


@dataclass
class SessionContext:
    """Conversation session context"""
    session_id: str
    turns: List[ConversationTurn]
    active_entities: Dict[str, any]  # Entities carried across turns
    topic_thread: List[str]  # Topic progression
    created_at: str
    last_updated: str


class ContextInjector:
    """
    Manages conversation context and injects it into queries
    for maintaining coherence across turns.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize context injector.
        
        Args:
            max_history: Maximum conversation turns to maintain
        """
        self.max_history = max_history
        self.sessions = {}  # session_id -> SessionContext
        
        logger.info(f"ContextInjector initialized with max_history={max_history}")
    
    def inject_context(self, query: str, 
                      session_id: str = "default",
                      entities: Dict[str, List] = None,
                      intent: Dict = None) -> Dict[str, any]:
        """
        Inject context from conversation history into current query.
        
        Args:
            query: Current query
            session_id: Session identifier
            entities: Entities from current query
            intent: Intent from current query
            
        Returns:
            Context-enriched query package
        """
        # Get or create session
        session = self._get_or_create_session(session_id)
        
        # Resolve references (pronouns, "it", "that", "previous")
        resolved_query = self._resolve_references(query, session)
        
        # Carry forward entities from previous turns
        enriched_entities = self._enrich_entities(entities or {}, session)
        
        # Build context summary
        context_summary = self._build_context_summary(session)
        
        # Update session with current turn
        self._update_session(session, query, entities, intent)
        
        result = {
            "query": query,
            "resolved_query": resolved_query,
            "entities": enriched_entities,
            "context_summary": context_summary,
            "session_id": session_id,
            "turn_number": len(session.turns),
            "has_context": len(session.turns) > 1
        }
        
        logger.debug(f"Injected context for session {session_id}, turn {len(session.turns)}")
        return result
    
    def _get_or_create_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionContext(
                session_id=session_id,
                turns=[],
                active_entities={},
                topic_thread=[],
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
        return self.sessions[session_id]
    
    def _resolve_references(self, query: str, session: SessionContext) -> str:
        """
        Resolve references to previous entities/topics.
        
        Handles:
        - "What about it?" -> "What about [previous entity]?"
        - "Show me that" -> "Show me [previous topic]"
        - "The same district" -> "[previous district]"
        """
        resolved = query
        query_lower = query.lower()
        
        if len(session.turns) == 0:
            return resolved  # No history to resolve
        
        last_turn = session.turns[-1]
        
        # Reference patterns
        reference_patterns = {
            r'\bit\b': self._get_last_entity_of_type(session, ["schemes", "districts", "go_numbers"]),
            r'\bthat\b': self._get_last_entity_of_type(session, ["schemes", "districts"]),
            r'\bthis\b': self._get_last_entity_of_type(session, ["schemes", "districts"]),
            r'\bthe same\b': self._get_last_entity_value(session),
            r'\bprevious\b': self._get_last_topic(session),
        }
        
        for pattern, replacement in reference_patterns.items():
            if replacement and pattern in query_lower:
                resolved = resolved.replace(pattern, replacement, 1)
        
        # Handle "also" or "too" - carry forward previous context
        if any(word in query_lower for word in ["also", "too", "as well"]):
            last_topic = self._get_last_topic(session)
            if last_topic:
                resolved = f"{resolved} ({last_topic})"
        
        return resolved
    
    def _get_last_entity_of_type(self, session: SessionContext, 
                                 entity_types: List[str]) -> Optional[str]:
        """Get the most recent entity of specified types"""
        for turn in reversed(session.turns):
            for entity_type in entity_types:
                if turn.entities.get(entity_type):
                    entities = turn.entities[entity_type]
                    if entities:
                        # Get first entity's text or canonical form
                        entity = entities[0]
                        if isinstance(entity, dict):
                            return entity.get("canonical", entity.get("text", ""))
                        else:
                            return str(entity)
        return None
    
    def _get_last_entity_value(self, session: SessionContext) -> Optional[str]:
        """Get any last entity value"""
        if not session.turns:
            return None
        
        last_turn = session.turns[-1]
        for entity_list in last_turn.entities.values():
            if entity_list:
                entity = entity_list[0]
                if isinstance(entity, dict):
                    return entity.get("canonical", entity.get("text", ""))
                else:
                    return str(entity)
        return None
    
    def _get_last_topic(self, session: SessionContext) -> Optional[str]:
        """Get the last discussed topic"""
        if session.topic_thread:
            return session.topic_thread[-1]
        return None
    
    def _enrich_entities(self, current_entities: Dict[str, List], 
                        session: SessionContext) -> Dict[str, List]:
        """
        Enrich current entities with context from session.
        Carries forward relevant entities if not specified in current query.
        """
        enriched = current_entities.copy()
        
        # If certain entity types are missing but were mentioned before,
        # consider carrying them forward (with lower confidence)
        carryforward_types = ["districts", "dates"]
        
        for entity_type in carryforward_types:
            if not enriched.get(entity_type) and session.active_entities.get(entity_type):
                # Carry forward from session with metadata
                enriched[entity_type] = [{
                    "text": session.active_entities[entity_type],
                    "canonical": session.active_entities[entity_type],
                    "confidence": 0.7,  # Lower confidence for carried-forward
                    "source": "context"
                }]
        
        return enriched
    
    def _build_context_summary(self, session: SessionContext) -> str:
        """Build natural language summary of conversation context"""
        if not session.turns:
            return "No previous context"
        
        # Last N turns
        recent_turns = session.turns[-self.max_history:]
        
        summary_parts = []
        
        # Add topic thread
        if session.topic_thread:
            topics = " -> ".join(session.topic_thread[-3:])
            summary_parts.append(f"Topic progression: {topics}")
        
        # Add active entities
        if session.active_entities:
            entities_str = ", ".join([f"{k}: {v}" for k, v in session.active_entities.items()])
            summary_parts.append(f"Active context: {entities_str}")
        
        # Add recent queries
        if recent_turns:
            recent_queries = [turn.query for turn in recent_turns[-2:]]
            summary_parts.append(f"Recent queries: {'; '.join(recent_queries)}")
        
        return " | ".join(summary_parts) if summary_parts else "No context"
    
    def _update_session(self, session: SessionContext, 
                       query: str,
                       entities: Dict[str, List] = None,
                       intent: Dict = None):
        """Update session with current turn"""
        # Add turn
        turn = ConversationTurn(
            query=query,
            entities=entities or {},
            intent=intent or {},
            timestamp=datetime.now().isoformat()
        )
        session.turns.append(turn)
        
        # Keep only last N turns
        if len(session.turns) > self.max_history:
            session.turns = session.turns[-self.max_history:]
        
        # Update active entities
        if entities:
            for entity_type, entity_list in entities.items():
                if entity_list:
                    # Take first entity as active
                    entity = entity_list[0]
                    if isinstance(entity, dict):
                        session.active_entities[entity_type] = entity.get("canonical", entity.get("text", ""))
                    else:
                        session.active_entities[entity_type] = str(entity)
        
        # Update topic thread
        if intent and intent.get("primary"):
            primary_intent = intent["primary"]
            intent_name = primary_intent if isinstance(primary_intent, str) else primary_intent.get("name", "unknown")
            
            # Add to topic thread if new topic
            if not session.topic_thread or session.topic_thread[-1] != intent_name:
                session.topic_thread.append(intent_name)
                
                # Keep thread length manageable
                if len(session.topic_thread) > 5:
                    session.topic_thread = session.topic_thread[-5:]
        
        # Update timestamp
        session.last_updated = datetime.now().isoformat()
    
    def add_response(self, session_id: str, response_summary: str):
        """Add response summary to last turn"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.turns:
                session.turns[-1].response_summary = response_summary
    
    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Get current session state"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return asdict(session)
        return None
    
    def clear_session(self, session_id: str):
        """Clear session history"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")


# Convenience function for backwards compatibility
def inject_context(query: str, conversation_history: list = None) -> str:
    """Add context from previous messages (backwards compatible)"""
    if not conversation_history:
        return query
    
    context = "Previous conversation: " + " ".join(conversation_history[-3:])
    return f"{context}\n\nCurrent query: {query}"


# Main API function
def process_context_injection(query: str,
                              session_id: str = "default",
                              entities: Dict[str, List] = None,
                              intent: Dict = None) -> Dict[str, any]:
    """
    Process query with context injection.
    
    Args:
        query: Current query
        session_id: Session identifier
        entities: Extracted entities
        intent: Classified intent
        
    Returns:
        Context-enriched query package
    """
    injector = ContextInjector()
    return injector.inject_context(query, session_id, entities, intent)
