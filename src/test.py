#!/usr/bin/env python3
"""
Test script that sends 4 concurrent requests to /api/query every 3 seconds
"""

import asyncio
import httpx
import time
from datetime import datetime
import json


class ConcurrentTester:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

        # Different test messages to make requests more interesting
        self.test_messages = [
            "What is the weather like in Tokyo?",
            "Tell me a fun fact about beavers",
            "How does photosynthesis work?",
            "What is the capital of Morocco?",
        ]

    async def send_single_request(self, message: str, request_id: int):
        """Send a single request to the API"""
        try:
            start_time = time.time()

            payload = {"message": message, "force_refresh": False}

            response = await self.client.post(f"{self.base_url}/api/query", json=payload)

            end_time = time.time()
            duration = end_time - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Request {request_id}: {duration:.2f}s - {message[:30]}...")
                print(f"   Response: {result.get('response', 'No response')[:50]}...")
                print(f"   Source: {result.get('metadata', {}).get('source', 'unknown')}")
                return True, result
            else:
                print(f"❌ Request {request_id}: Error {response.status_code} - {message[:30]}...")
                return False, {"error": response.text}

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"💥 Request {request_id}: Exception after {duration:.2f}s - {str(e)}")
            return False, {"error": str(e)}

    async def send_concurrent_batch(self, batch_number: int):
        """Send 4 concurrent requests"""
        print(f"\n🚀 Batch {batch_number} - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)

        # Create 4 concurrent tasks
        tasks = []
        for i in range(4):
            message = self.test_messages[i % len(self.test_messages)]
            task = self.send_single_request(message, i + 1)
            tasks.append(task)

        # Execute all requests concurrently
        batch_start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_end = time.time()

        # Summary
        successful = sum(
            1 for success, _ in results if success and not isinstance(success, Exception)
        )
        print(f"\n📊 Batch {batch_number} Summary:")
        print(f"   Total time: {batch_end - batch_start:.2f}s")
        print(f"   Successful: {successful}/4")

        return results

    async def check_server_health(self):
        """Check if the server is running"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"🟢 Server is healthy: {health_data}")
                return True
            else:
                print(f"🟡 Server responded with status {response.status_code}")
                return False
        except Exception as e:
            print(f"🔴 Server is not reachable: {str(e)}")
            return False

    async def run_continuous_test(self, interval_seconds: int = 3):
        """Run the continuous testing loop"""
        print("🎯 Starting concurrent request testing...")
        print(f"📡 Target: {self.base_url}/api/query")
        print(f"⏱️  Interval: {interval_seconds} seconds")
        print(f"🔢 Concurrent requests per batch: 4")

        # Check server health first
        if not await self.check_server_health():
            print("❌ Server health check failed. Please ensure the server is running.")
            return

        batch_number = 1

        try:
            while True:
                await self.send_concurrent_batch(batch_number)
                batch_number += 1

                # Wait before next batch
                print(f"⏳ Waiting {interval_seconds} seconds before next batch...")
                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n🛑 Testing stopped by user")
        except Exception as e:
            print(f"\n💥 Unexpected error: {str(e)}")
        finally:
            await self.client.aclose()


async def main():
    """Main function"""
    tester = ConcurrentTester()
    await tester.run_continuous_test(interval_seconds=3)


if __name__ == "__main__":
    # Run the test
    print("🔥 Concurrent API Test Script")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    asyncio.run(main())
