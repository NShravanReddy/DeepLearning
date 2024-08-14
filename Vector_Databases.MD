  Reason #1: Over 80% of the data produced is unstructured!
  Example: Social Media, Image, Video, Audio
  Reason#2: LLM lacks Long-Term Memory
  Vector DBs provide the ability to store & retrieve data for LLMs

### vector embeddings
like vector means 2d matrix


Types of Vector Embeddings
Word Embeddings:
• Techniques: Word2 Vec, GloVe, FastText
• Purpose: Capture semantic relationships and contextual information.
Sentence Embeddings:
• Models: Universal Sentence Encoder (USE), Skip Thought
• Purpose: Represent overall meaning and context of sentences.
Document Embeddings:
• Techniques: Doc2Vec, Paragraph Vectors
• Purpose: Capture semantic information and context of entire documents.
Image Embeddings:
• Techniques: CNNs, ResNet, VGG
• Purpose: Capture visual features for tasks like classification and object detection.


#####
Data type image, document, audio pass this through pre trained model that pre trained model and that pretriend model will prvide vector embeddings
If there are millions of dcumets we need to index them amd store vector embeddings i database in an optimized way, 
When the user provides the query it will conveted to vector ebdeddings by a pre trained model and that vector embedding will be sent into to the vector database to search for the matching vector embedding and it will provide the results. That result will be semantic 

Provides the ability to compare multiple things (semantically) at the same time • Helps machine learning models remember past data better, making them more useful for search,recommendations, and text generation

#### What is Vector databases?
• A vector database indexes and stores vector embeddings, for fast search and optimized storage
• Provides the ability to compare multiple things (semantically) at the same time
• Helps machine learning models remember past data better, making them more useful for search, recommendations, and text generation
