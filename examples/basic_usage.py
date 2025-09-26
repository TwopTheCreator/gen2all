from gen2all import Gen2AllClient
import asyncio
import time


def basic_generation_example():
    print("🚀 Gen2All Basic Generation Example")
    print("=" * 50)
    
    client = Gen2AllClient(
        api_key="your_api_key_here",
        base_url="http://localhost:8000"
    )
    
    prompt = "Write a creative story about a robot discovering emotions"
    
    print(f"📝 Generating response for: {prompt[:50]}...")
    start_time = time.time()
    
    response = client.generate(
        prompt=prompt,
        max_length=500,
        temperature=0.8,
        top_p=0.9
    )
    
    end_time = time.time()
    
    print(f"✅ Generation completed in {end_time - start_time:.2f} seconds")
    print(f"📊 Tokens generated: {response['token_count']}")
    print("\n📖 Generated text:")
    print("-" * 50)
    print(response['generated_text'])
    print("-" * 50)


def conversation_example():
    print("\n💬 Gen2All Conversation Example")
    print("=" * 50)
    
    client = Gen2AllClient(
        api_key="your_api_key_here",
        base_url="http://localhost:8000"
    )
    
    context_id = client.create_context(
        system_message="You are a helpful AI assistant specialized in technology and programming."
    )
    
    questions = [
        "What is machine learning?",
        "Can you explain neural networks?",
        "How do transformers work in AI?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🤔 Question {i}: {question}")
        
        response = client.chat(
            message=question,
            context_id=context_id,
            max_length=300,
            temperature=0.7
        )
        
        print(f"🤖 Response: {response['generated_text']}")
        print(f"📊 Tokens: {response['token_count']}")
    
    context_info = client.get_context_info(context_id)
    print(f"\n📈 Context Summary:")
    print(f"   Messages: {context_info['message_count']}")
    print(f"   Total tokens: {context_info['token_count']}")


def batch_processing_example():
    print("\n📦 Gen2All Batch Processing Example")
    print("=" * 50)
    
    client = Gen2AllClient(
        api_key="your_api_key_here",
        base_url="http://localhost:8000"
    )
    
    prompts = [
        "Summarize the benefits of renewable energy",
        "Explain quantum computing in simple terms",
        "Describe the future of artificial intelligence",
        "What are the challenges in space exploration?",
        "How does blockchain technology work?"
    ]
    
    print(f"🔄 Processing {len(prompts)} prompts in batch...")
    start_time = time.time()
    
    results = client.batch_generate(
        prompts=prompts,
        generation_config={
            'max_length': 200,
            'temperature': 0.8,
            'top_k': 50
        },
        parallel_processing=True
    )
    
    end_time = time.time()
    
    print(f"✅ Batch processing completed in {end_time - start_time:.2f} seconds")
    
    total_tokens = sum(result['token_count'] for result in results if result['success'])
    successful = sum(1 for result in results if result['success'])
    
    print(f"📊 Results: {successful}/{len(results)} successful")
    print(f"📊 Total tokens generated: {total_tokens}")
    
    for i, result in enumerate(results):
        if result['success']:
            print(f"\n📝 Response {i+1}:")
            print(f"   {result['generated_text'][:100]}...")
            print(f"   Tokens: {result['token_count']}")


def specialized_tasks_example():
    print("\n🎯 Gen2All Specialized Tasks Example")
    print("=" * 50)
    
    client = Gen2AllClient(
        api_key="your_api_key_here",
        base_url="http://localhost:8000"
    )
    
    # Code completion
    print("\n💻 Code Completion:")
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    # Complete this function
"""
    
    completion = client.code_completion(code, language="Python")
    print(f"✅ Completed code:\n{completion}")
    
    # Translation
    print("\n🌍 Translation:")
    text = "Hello, how are you today? I hope you're having a great day!"
    translation = client.translate(text, "Spanish")
    print(f"🇪🇸 Spanish translation: {translation}")
    
    # Q&A with context
    print("\n❓ Question Answering:")
    context = "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has a vast ecosystem of libraries."
    question = "What are the main characteristics of Python?"
    answer = client.answer_question(question, context)
    print(f"🤖 Answer: {answer}")
    
    # Creative writing
    print("\n✍️ Creative Writing:")
    prompt = "A mysterious door appears in a busy city street"
    story = client.creative_writing(prompt, style="science fiction")
    print(f"📚 Story excerpt:\n{story[:200]}...")
    
    # Summarization
    print("\n📄 Summarization:")
    long_text = """
    Artificial Intelligence (AI) represents one of the most significant technological advances of our time. 
    It encompasses machine learning, deep learning, natural language processing, computer vision, and robotics. 
    AI systems can process vast amounts of data, recognize patterns, make predictions, and even create content. 
    The applications are endless: from healthcare diagnosis to autonomous vehicles, from financial trading to 
    entertainment recommendation systems. However, AI also raises important questions about ethics, privacy, 
    job displacement, and the future of human-machine interaction. As AI continues to evolve, it's crucial 
    to develop it responsibly and ensure it benefits all of humanity.
    """
    
    summary = client.summarize(long_text, max_length=100)
    print(f"📋 Summary: {summary}")


async def async_example():
    print("\n⚡ Gen2All Async Processing Example")
    print("=" * 50)
    
    client = Gen2AllClient(
        api_key="your_api_key_here",
        base_url="http://localhost:8000"
    )
    
    prompts = [
        "Explain photosynthesis",
        "Describe the water cycle",
        "What causes earthquakes?",
        "How do vaccines work?"
    ]
    
    print(f"🚀 Starting {len(prompts)} async generations...")
    start_time = time.time()
    
    # Submit all requests asynchronously
    futures = []
    for prompt in prompts:
        future = client.generate_async(
            prompt=prompt,
            max_length=150,
            temperature=0.7
        )
        futures.append((prompt, future))
    
    # Collect results as they complete
    for prompt, future in futures:
        result = future.result()
        print(f"✅ '{prompt[:30]}...' -> {result['token_count']} tokens")
    
    end_time = time.time()
    print(f"🏁 All async operations completed in {end_time - start_time:.2f} seconds")


def monitoring_example():
    print("\n📊 Gen2All Monitoring Example")
    print("=" * 50)
    
    client = Gen2AllClient(
        api_key="your_api_key_here",
        base_url="http://localhost:8000"
    )
    
    # Check health
    if client.health_check():
        print("✅ Gen2All service is healthy")
    else:
        print("❌ Gen2All service is not responding")
        return
    
    # Get comprehensive stats
    stats = client.get_stats()
    
    print("\n🖥️ Server Statistics:")
    server_stats = stats['server_stats']
    print(f"   Requests processed: {server_stats.get('requests_processed', 0)}")
    print(f"   Tokens generated: {server_stats.get('tokens_generated', 0)}")
    print(f"   Active contexts: {server_stats.get('contexts_created', 0)}")
    print(f"   Cache hit rate: {server_stats.get('cache_hit_rate', 0):.2%}")
    print(f"   Uptime: {server_stats.get('uptime_seconds', 0):.0f} seconds")
    
    system_stats = server_stats.get('system_stats', {})
    print(f"\n💻 System Resources:")
    print(f"   CPU usage: {system_stats.get('cpu_percent', 0):.1f}%")
    print(f"   Memory usage: {system_stats.get('memory_percent', 0):.1f}%")
    
    print("\n📱 Client Statistics:")
    client_stats = stats['client_stats']
    print(f"   Client requests: {client_stats.get('requests_made', 0)}")
    print(f"   Client tokens: {client_stats.get('tokens_generated', 0)}")
    print(f"   Client errors: {client_stats.get('errors', 0)}")
    print(f"   Client uptime: {client_stats.get('uptime_seconds', 0):.0f} seconds")


def main():
    print("🤖 Gen2All Python Client Examples")
    print("=" * 60)
    print("Make sure Gen2All server is running on localhost:8000")
    print("Update 'your_api_key_here' with your actual API key")
    print("=" * 60)
    
    try:
        # Run basic examples
        basic_generation_example()
        conversation_example()
        batch_processing_example()
        specialized_tasks_example()
        
        # Run async example
        asyncio.run(async_example())
        
        # Run monitoring example
        monitoring_example()
        
        print("\n🎉 All examples completed successfully!")
        print("🚀 Start building amazing AI applications with Gen2All!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure:")
        print("1. Gen2All server is running (python -m gen2all.api.server)")
        print("2. You have a valid API key (run APIGetter.py)")
        print("3. All dependencies are installed (pip install -e .)")


if __name__ == "__main__":
    main()