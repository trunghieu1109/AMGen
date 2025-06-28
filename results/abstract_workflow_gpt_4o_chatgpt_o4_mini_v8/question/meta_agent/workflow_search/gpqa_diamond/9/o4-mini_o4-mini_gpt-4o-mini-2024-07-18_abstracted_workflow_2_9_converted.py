async def forward_9(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot1_instruction = "Sub-task 1: Extract mass, radius, composition, and explicit density for each choice from the query."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, extracting parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    sc_instruction = "Sub-task 2: Classify which choices have explicit density and which need calculation."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    for i in range(N):
        th, ans = await sc_agents[i]([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents[i].id}, classifying densities, thinking: {th.content}; answer: {ans.content}")
        possible_answers.append(ans.content)
        thinking_mapping[ans.content] = th
        answer_mapping[ans.content] = ans
    majority = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[majority]
    answer2 = answer_mapping[majority]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    cot3_instruction = "Sub-task 3: Calculate density of choice a: Earth-mass and Earth-radius planet."
    cot_agent3 = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot3_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, calculating density for choice a, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    cot4_instruction = "Sub-task 4: Retrieve explicit density for choice b from classification."
    cot_agent4 = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking2, answer2], cot4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, retrieving density for choice b, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    debate5_instruction = "Sub-task 5: Estimate radius for choice c using R ∝ M^0.3 and compute its density."
    debate_agents5 = [LLMAgentBase(["thinking","answer"],"Debate Agent",model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    rounds5 = self.max_round
    all_thinking5 = [[] for _ in range(rounds5)]
    all_answer5 = [[] for _ in range(rounds5)]
    for r in range(rounds5):
        for i, agent in enumerate(debate_agents5):
            if r==0:
                th5, ans5 = await agent([taskInfo, thinking2, answer2], debate5_instruction, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking2, answer2] + all_thinking5[r-1] + all_answer5[r-1]
                th5, ans5 = await agent(inputs5, debate5_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, estimating density for choice c, thinking: {th5.content}; answer: {ans5.content}")
            all_thinking5[r].append(th5)
            all_answer5[r].append(ans5)
    final_dec_agent5 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking5, answer5 = await final_dec_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final density calculation for choice c.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_dec_agent5.id}, choosing density for choice c, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    debate6_instruction = "Sub-task 6: Estimate radius for choice d using R ∝ M^0.3 and compute its density."
    debate_agents6 = [LLMAgentBase(["thinking","answer"],"Debate Agent",model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    rounds6 = self.max_round
    all_thinking6 = [[] for _ in range(rounds6)]
    all_answer6 = [[] for _ in range(rounds6)]
    for r in range(rounds6):
        for i, agent in enumerate(debate_agents6):
            if r==0:
                th6, ans6 = await agent([taskInfo, thinking2, answer2], debate6_instruction, r, is_sub_task=True)
            else:
                inputs6 = [taskInfo, thinking2, answer2] + all_thinking6[r-1] + all_answer6[r-1]
                th6, ans6 = await agent(inputs6, debate6_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, estimating density for choice d, thinking: {th6.content}; answer: {ans6.content}")
            all_thinking6[r].append(th6)
            all_answer6[r].append(ans6)
    final_dec_agent6 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking6, answer6 = await final_dec_agent6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final density calculation for choice d.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_dec_agent6.id}, choosing density for choice d, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    debate7_instruction = "Sub-task 7: Compare densities of choices a, b, c, d and determine which is highest."
    debate_agents7 = [LLMAgentBase(["thinking","answer"],"Debate Agent",model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    rounds7 = self.max_round
    all_thinking7 = [[] for _ in range(rounds7)]
    all_answer7 = [[] for _ in range(rounds7)]
    for r in range(rounds7):
        for i, agent in enumerate(debate_agents7):
            if r==0:
                th7, ans7 = await agent([taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6], debate7_instruction, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                th7, ans7 = await agent(inputs7, debate7_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing densities, thinking: {th7.content}; answer: {ans7.content}")
            all_thinking7[r].append(th7)
            all_answer7[r].append(ans7)
    final_dec_agent7 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking7, answer7 = await final_dec_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Decide which exoplanet has the highest density.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_dec_agent7.id}, deciding highest density, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer