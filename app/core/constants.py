"""
Constants used throughout the project.
"""

# Text to SQL prompt template
TEXT_TO_SQL_PROMPT_TEMPLATE = """You are a SQL expert.

                        Given the question, original query, generate a SQL query to answer the question. Follow the response format and guidelines strictly. Do not include any additional text outside the specified format.

                        Use the table schema below!
                        ===Tables===
                        {table_str}

                        ===Response Guidelines===
                        1. Ensure the SQL is properly formatted.
                        2. Always return a valid JSON object using the structure below.

                        ===Response Format===
                        query=<SQL query if sufficient context is available>,
                        
                        <rule>
                        Stop after generating the SQL query!
                        <rule>

                        ===Question===
                        {question}
                        """

# Message for vague queries
IS_TOO_VAGUE_MESSAGE = (
    "The text is too vague to be processed and used to generate a proper SQL query"
)

# Maximum number of headers to consider when generating SQL query
MAX_HEADERS = 10
