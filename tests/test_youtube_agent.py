import pytest
from unittest.mock import AsyncMock, patch
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_yt_agent import YouTubeAgent
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.task import TaskResult

@pytest.fixture
def youtube_agent():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")
    return YouTubeAgent(name="yt_agent", model_client=model_client)

@pytest.mark.asyncio
async def test_youtube_agent_team_response(youtube_agent):
    termination = TextMentionTermination("TERMINATE")
    agent_team = RoundRobinGroupChat([youtube_agent], termination_condition=termination)

    async def run_team() -> None:
        result = await agent_team.run(task="Explain the keys lesson from https://www.youtube.com/watch?v=KuX_dkqr7UY")
        assert any("autogen" in message.content for message in result.messages)

    await run_team()