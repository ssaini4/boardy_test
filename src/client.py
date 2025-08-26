#!/usr/bin/env python3
"""
Terminal client for the FastAPI chatbot server
"""

import httpx
import asyncio
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

console = Console()
app = typer.Typer()


class ChatbotClient:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def health_check(self):
        """Check if the server is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}

    async def send_message(
        self, message: str, system_prompt: str = None, include_embedding: bool = False
    ):
        """Send a message to the chatbot"""
        try:
            endpoint = "/api/query"
            payload = {
                "message": message,
                "system_prompt": system_prompt or "You are a helpful AI assistant.",
                "max_tokens": 500,
                "temperature": 0.7,
            }

            response = await self.client.post(f"{self.base_url}{endpoint}", json=payload)

            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {
                    "error": f"Server error: {response.status_code}",
                    "detail": response.text,
                }

        except Exception as e:
            return False, {"error": str(e)}

    async def close(self):
        """Close the client"""
        await self.client.aclose()


def display_welcome():
    """Display welcome message"""
    welcome_text = """
    🤖 FastAPI Chatbot Terminal Client
    
    Commands:
    • Type your message and press Enter to chat
    • Type 'quit' or 'exit' to leave
    • Type 'help' for more commands
    • Type 'health' to check server status

    """

    console.print(Panel(welcome_text, title="Welcome", style="cyan"))


def display_help():
    """Display help information"""
    help_text = """
    Available Commands:
    
    • Normal message: Just type your message
    • quit/exit: Exit the client
    • health: Check server health
    • system <prompt>: Set system prompt
    • help: Show this help
    """

    console.print(Panel(help_text, title="Help", style="yellow"))


async def interactive_mode(base_url: str):
    """Run interactive chat mode"""
    client = ChatbotClient(base_url)
    system_prompt = "You are a helpful AI assistant."

    try:
        # Check server health
        console.print("🔍 Checking server connection...", style="yellow")
        is_healthy, health_data = await client.health_check()

        if not is_healthy:
            console.print(
                f"❌ Cannot connect to server: {health_data.get('error', 'Unknown error')}",
                style="red",
            )
            return

        console.print("✅ Server is healthy!", style="green")
        console.print(f"Server status: {health_data}", style="dim")

        display_welcome()

        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.lower() in ["quit", "exit"]:
                    console.print("👋 Goodbye!", style="cyan")
                    break

                elif user_input.lower() == "help":
                    display_help()
                    continue

                elif user_input.lower() == "health":
                    is_healthy, health_data = await client.health_check()
                    if is_healthy:
                        console.print("✅ Server is healthy!", style="green")
                        console.print(json.dumps(health_data, indent=2), style="dim")
                    else:
                        console.print(f"❌ Server health check failed: {health_data}", style="red")
                    continue

                elif user_input.lower().startswith("embed "):
                    text_to_embed = user_input[6:].strip()
                    if text_to_embed:
                        console.print("🔄 Generating embedding...", style="yellow")
                        success, result = await client.get_embedding(text_to_embed)

                        if success:
                            embedding_info = f"""
Model: {result['model']}
Dimensions: {result['dimensions']}
Embedding: {result['embedding'][:5]}... (showing first 5 values)
                            """
                            console.print(
                                Panel(
                                    embedding_info,
                                    title="Embedding Result",
                                    style="green",
                                )
                            )
                        else:
                            console.print(
                                f"❌ Error: {result.get('error', 'Unknown error')}",
                                style="red",
                            )
                    continue

                elif user_input.lower().startswith("system "):
                    system_prompt = user_input[7:].strip()
                    console.print(f"✅ System prompt updated: {system_prompt}", style="green")
                    continue

                # Send regular chat message
                console.print("🤔 Thinking...", style="yellow")
                success, result = await client.send_message(user_input, system_prompt)

                if success:
                    response_text = result.get("response", "No response")

                    # Display the response
                    console.print("\n[bold green]Assistant[/bold green]:")
                    console.print(
                        Panel(Markdown(response_text), style="green"),
                        Panel(
                            Markdown(result.get("metadata", "No metadata")["source"]),
                            title="Source",
                            style="dim",
                        ),
                    )

                    # If embedding is included, show some info
                    if "embedding" in result and result["embedding"]:
                        embedding_info = f"Embedding generated using {result.get('embedding_model', 'unknown')} ({len(result['embedding'])} dimensions)"
                        console.print(f"📊 {embedding_info}", style="dim")

                else:
                    console.print(f"❌ Error: {result.get('error', 'Unknown error')}", style="red")
                    if "detail" in result:
                        console.print(f"Details: {result['detail']}", style="dim red")

            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="cyan")
                break
            except Exception as e:
                console.print(f"❌ Unexpected error: {e}", style="red")

    finally:
        await client.close()


@app.command()
def chat(
    server_url: str = typer.Option("http://localhost:3000", help="FastAPI server URL"),
    message: str = typer.Option(None, help="Send a single message instead of interactive mode"),
):
    """
    Terminal client for the FastAPI chatbot server
    """
    if message:
        # Single message mode
        async def send_single_message():
            client = ChatbotClient(server_url)
            try:
                console.print(f"Sending: {message}")
                success, result = await client.send_message(message)

                if success:
                    console.print("\nResponse:")
                    console.print(
                        Panel(
                            Markdown(result.get("response", "No response")),
                            style="green",
                        ),
                        # Panel(
                        #     Markdown(result.get("metadata", "No metadata")["source"]),
                        #     title="Source",
                        #     style="pink",
                        # ),
                    )
                else:
                    console.print(f"Error: {result.get('error', 'Unknown error')}", style="red")
            finally:
                await client.close()

        asyncio.run(send_single_message())
    else:
        # Interactive mode
        asyncio.run(interactive_mode(server_url))


if __name__ == "__main__":
    app()
