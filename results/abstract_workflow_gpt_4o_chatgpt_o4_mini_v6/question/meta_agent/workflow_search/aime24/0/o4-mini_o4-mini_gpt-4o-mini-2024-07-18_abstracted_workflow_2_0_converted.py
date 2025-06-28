async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], "Sub-task 1: Identify all given numerical data from the problem statement: total distance 9 km, times 4 h and 2 h 24 min, speed expressions s, s+2, s+0.5.", is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], "Sub-task 2: Identify unknown variables: Aya’s base walking speed s (km/h) and coffee-shop stop time t (minutes).", is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying unknowns, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2], "Sub-task 3: Convert times: 4 h → 4 - t/60 h, and 2 h 24 min = 2.4 h → 2.4 - t/60 h.", is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, converting times, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], "Sub-task 4: Form equation 9/s = 4 - t/60 for the first scenario.", is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, writing eq1, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking3, answer3], "Sub-task 5: Form equation 9/(s+2) = 2.4 - t/60 for the second scenario.", is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, writing eq2, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 6: Solve the system 9/s = 4 - t/60 and 9/(s+2) = 2.4 - t/60 for s and t."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    for i in range(N):
        thinking6_i, answer6_i = await cot_agents[i]([taskInfo, thinking4, answer4, thinking5, answer5], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, solving system, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_answers.append(answer6_i.content)
    counts = Counter(possible_answers)
    selected6 = counts.most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 6 output: answer - {selected6}")
    print("Step 6: ", sub_tasks[-1])

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent([taskInfo, selected6], "Sub-task 7: Formulate walking time when speed is s+0.5: time = 9/(s+0.5) hours.", is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, forming time expr, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent([taskInfo, selected6, thinking7, answer7], "Sub-task 8: Convert 9/(s+0.5) hours to minutes, add t, and report total minutes.", is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing final time, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer