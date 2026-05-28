class QueryClassifier:
    """
    Intelligent Query Classifier
    """

    def __init__(self):
        pass

    def classify(self, query):
        """
        Detect query type.

        Args:
            query (str): User query

        Returns:
            str: Query category
        """

        query = query.lower().strip()

        # ---------------------------------------------------
        # Greeting Queries
        # ---------------------------------------------------

        greeting_keywords = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good evening",
            "how are you",
            "how r u",
            "yo",
            "hola"
        ]

        # ---------------------------------------------------
        # Appreciation / Thanks
        # ---------------------------------------------------

        thanks_keywords = [
            "thank you",
            "thanks",
            "thx",
            "ty",
            "great",
            "nice",
            "awesome"
        ]

        # ---------------------------------------------------
        # Analytical Queries
        # ---------------------------------------------------

        analytical_keywords = [
            "why",
            "how",
            "analyze",
            "compare",
            "difference",
            "impact",
            "advantages",
            "disadvantages",
            "explain deeply",
            "elaborate"
        ]

        # ---------------------------------------------------
        # Opinion Queries
        # ---------------------------------------------------

        opinion_keywords = [
            "opinion",
            "suggest",
            "recommend",
            "best",
            "should i",
            "which is better"
        ]

        # ---------------------------------------------------
        # Factual Queries
        # ---------------------------------------------------

        factual_keywords = [
            "what",
            "when",
            "where",
            "who",
            "define",
            "meaning"
        ]
        # Summary Queries
        summary_keywords = [
            "summary",
            "summarize",
            "overview",
            "what is this pdf about",
            "about the pdf",
            "about this document"
        ]
        # ---------------------------------------------------
        # Query Classification Logic
        # ---------------------------------------------------

        # Greeting Detection
        if any(word in query for word in greeting_keywords):
            return "greeting"

        # Appreciation Detection
        elif any(word in query for word in thanks_keywords):
            return "thanks"

        # Analytical Queries
        elif any(word in query for word in analytical_keywords):
            return "analytical"

        # Opinion Queries
        elif any(word in query for word in opinion_keywords):
            return "opinion"

        # Factual Queries
        elif any(word in query for word in factual_keywords):
            return "factual"

        elif any(word in query for word in summary_keywords):
            return "summary"
        
        # Default Contextual Query
        else:
            return "contextual"