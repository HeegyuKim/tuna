from eval.utils import load_model
import discord
from discord.ext import commands
from datetime import datetime
import pytz
import sys
import fire
import os
from collections import defaultdict

DEFAULT_SYSTEM_PROMPT = "당신은 아주대학교 iKnow Lab 챗봇입니다. 사용자에게 유용하고, 반복적이지 않고, 계속해서 새로운 정보가 담긴 답변을 제공해야합니다. 사용자의 질문에 유창한 한글로 답변해주세요."

class ChatBot(commands.Cog):
    def __init__(self, bot, model, max_turns: int = 4, system_prompt: str = None):
        self.bot = bot
        self.model = model
        self.conversation_history = defaultdict(list)
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    @commands.Cog.listener("on_message")
    async def answer_dm_or_mention(self, message):
        print(message)

        if message.author == self.bot.user:
            return
        
        if isinstance(message.channel, discord.channel.DMChannel):
            user = message.author.id
            prompt = message.content.replace(self.bot.user.mention, '').strip()
            reply = self.answer(user, prompt)
            await message.channel.send(reply)
            return
        
        elif self.user in message.mentions:
            user = message.author.id
            prompt = message.content.replace(self.bot.user.mention, '').strip()
            reply = self.answer(user, prompt)
            await message.reply(reply)

        await self.process_commands(message)


    @commands.command()
    async def chat(self, ctx, *, message):
        user_id = ctx.author.id
        
        response = self.answer(user_id, message)
        
        await ctx.send(response)

    @commands.command()
    async def clear(self, ctx):
        user_id = ctx.author.id
        self.conversation_history[user_id].clear()
        await ctx.send("대화 기록이 초기화되었습니다.")

    @commands.command()
    async def usage(self, ctx):
        help_text = """
        사용 가능한 명령어:
        !chat [메시지] - 봇과 대화를 시작합니다. (혹은 멘션)
        !clear - 대화 기록을 초기화합니다.
        !usage - 이 도움말을 표시합니다.
        """
        await ctx.send(help_text)

    def answer(self, user_id, prompt):
        history = self.conversation_history[user_id]
        if len(history) == 0:
            history = None

        if self.system_prompt:
            if history:
                history = [{"role": "system", "content": self.system_prompt}] + history
            else:
                history = [{"role": "system", "content": self.system_prompt}]

        output = self.model.generate(prompt, history, gen_args={"do_sample": True, "verbose": True})

        self.conversation_history[user_id].append({"content": prompt, "role": "user"})
        self.conversation_history[user_id].append({"content": output, "role": "assistant"})
        
        if len(self.conversation_history[user_id]) > self.max_turns * 2:
            self.conversation_history[user_id] = self.conversation_history[user_id][-(self.max_turns * 2):]

        return output


def main(
        model: str,
        token: str = os.environ.get("DISCORD_TOKEN"),
        chat_template: str = None,
        prompt_length: int = 2048,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token: str = None,
        batch_size: int = 1,
        max_turns: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    model_name = model


    model = load_model(
        model_name,
        prompt_length=prompt_length,
        max_length=prompt_length + max_new_tokens,
        gen_args={"temperature": temperature, "top_k": top_k, "top_p": top_p},
        chat_template=chat_template,
        eos_token=eos_token,
        batch_size=batch_size
    )
    model.compile(targets=["sample"])

    class BotSetup(commands.Bot):
        def __init__(self):
            self.to_zone = pytz.timezone('Asia/Seoul')  # 원하는 시간대로 설정
            intents = discord.Intents.default()
            intents.message_content = True
            intents.dm_messages = True
            
            super().__init__(command_prefix='!', intents=intents)

        async def setup_hook(self):
            await self.add_cog(ChatBot(self, model, max_turns, system_prompt))
            print(f'{self.user}로 로그인했습니다!')

        async def on_command_error(self, ctx, error):
            if isinstance(error, commands.CommandNotFound):
                await ctx.send("알 수 없는 명령어입니다. !help를 입력하여 사용 가능한 명령어를 확인하세요.")
            else:
                await ctx.send(f"오류가 발생했습니다: {str(error)}")

        async def on_ready(self):
            print(f'{self.user}로 로그인했습니다!')


    bot = BotSetup()
    bot.run(token)

if __name__ == '__main__':
    fire.Fire(main)