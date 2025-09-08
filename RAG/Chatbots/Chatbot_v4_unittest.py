import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.base import RunnableBinding
from Chatbot_v4 import ask_human, AskHuman  

class TestAskHumanNode(unittest.TestCase):

    @patch("Chatbot_v4.interrupt", return_value="Test Response")
    def test_ask_human_returns_tool_message(self, mock_interrupt):
        tool_call = {
            "name": "AskHuman",
            "id": "mock-tool-id",
            "args": {"question": "Please confirm?"}
        }

        mock_message = AIMessage(content="Waiting on you", tool_calls=[tool_call])
        state = {"messages": [mock_message]}

        result = ask_human(state)

        self.assertIn("messages", result)
        self.assertIsInstance(result["messages"][0], ToolMessage)
        self.assertEqual(result["messages"][0].tool_call_id, "mock-tool-id")
        self.assertEqual(result["messages"][0].content, "Test Response")
        mock_interrupt.assert_called_once_with("Please confirm?")


    @patch("Chatbot_v4.interrupt", return_value="Proceed with search")
    def test_ask_human_interrupt(self, mock_interrupt):
        tool_call = {
            "name": "AskHuman",
            "id": "abc123",
            "args": {"question": "Should I continue?"}
        }

        state = {
            "messages": [AIMessage(content="Need your input", tool_calls=[tool_call])]
        }

        result = ask_human(state)

        # Assertions
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        message = result["messages"][0]
        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(message.content, "Proceed with search")
        self.assertEqual(message.tool_call_id, "abc123")
        mock_interrupt.assert_called_once_with("Should I continue?")


from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from Chatbot_v4 import call_model

# class TestCallModelNode(unittest.TestCase):

#     @patch("Chatbot_v4.llm_with_tool.invoke")
#     def test_call_model_generates_ai_response(self, mock_invoke):
#         mock_response = AIMessage(content="New Delhi is the capital of India.")
#         mock_invoke.return_value = mock_response

#         state = {
#             "messages": [HumanMessage(content="What is the capital of India?")]
#         }

#         result = call_model(state)

#         self.assertIn("messages", result)
#         self.assertEqual(result["messages"][0].content, "New Delhi is the capital of India.")
#         mock_invoke.assert_called_once()

class TestCallModelNode(unittest.TestCase):

    @patch.object(RunnableBinding, "invoke")
    def test_call_model_generates_ai_response(self, mock_invoke):
        mock_response = AIMessage(content="New Delhi is the capital of India.")
        mock_invoke.return_value = mock_response

        state = {
            "messages": [HumanMessage(content="What is the capital of India?")]
        }

        result = call_model(state)

        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0].content, "New Delhi is the capital of India.")
        mock_invoke.assert_called_once_with(state["messages"])

from Chatbot_v4 import should_continue, State, END
from langchain_core.messages import AIMessage

class TestShouldContinueLogic(unittest.TestCase):

    def test_should_continue_end(self):
        state = {"messages": [AIMessage(content="All done!")]}
        self.assertEqual(should_continue(state), END)

    def test_should_continue_ask_human(self):
        tool_call = {"name": "AskHuman", "id": "123", "args": {"question": "Confirm?"}}
        state = {"messages": [AIMessage(content="Need human", tool_calls=[tool_call])]}
        self.assertEqual(should_continue(state), "ask_human")

    def test_should_continue_action(self):
        tool_call = {"name": "search", "id": "456", "args": {"query": "LangChain"}}
        state = {"messages": [AIMessage(content="Using search", tool_calls=[tool_call])]}
        self.assertEqual(should_continue(state), "action")
