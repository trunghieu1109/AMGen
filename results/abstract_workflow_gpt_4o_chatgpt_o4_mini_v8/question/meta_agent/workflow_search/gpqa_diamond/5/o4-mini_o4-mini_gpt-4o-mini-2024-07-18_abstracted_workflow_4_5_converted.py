async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Express polar coordinates (r, θ) in terms of Cartesian coordinates x and y using relations x = r*cos(θ), y = r*sin(θ)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, express polar coords, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Substitute x = r*cos(θ) and y = r*sin(θ) into V(r,θ) = 1/2 k r^2 + 3/2 k r^2 cos^2(θ) to get V(x,y)."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "SC-CoT Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    answers2 = []
    thinking2_map = {}
    answer2_map = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, substitution reasoning, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        answers2.append(answer2_i.content)
        thinking2_map[answer2_i.content] = thinking2_i
        answer2_map[answer2_i.content] = answer2_i
    best2 = Counter(answers2).most_common(1)[0][0]
    thinking2 = thinking2_map[best2]
    answer2 = answer2_map[best2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    cot3_instruction = "Sub-task 3: Simplify V(x,y) by collecting terms in x^2 and y^2 to express it as 1/2 k_x x^2 + 1/2 k_y y^2."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot3_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, simplify V(x,y), thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot4_instruction = "Sub-task 4: Identify effective spring constants k_x and k_y from the coefficients of x^2 and y^2 in the simplified potential."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, identify k_x and k_y, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    cot5_instruction = "Sub-task 5: Compute angular frequencies ω_x = sqrt(k_x/m) and ω_y = sqrt(k_y/m) using the effective spring constants."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot5_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, compute angular frequencies, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    cot6_instruction = "Sub-task 6: Write down the general energy eigenvalue expression E = ħ[ω_x(n_x + 1/2) + ω_y(n_y + 1/2)]."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot6_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, general energy expression, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    debate7_instruction = "Sub-task 7: Substitute ω_x and ω_y into the general energy formula and simplify to E = ħ sqrt(k/m) (a n_x + b n_y + c)."
    debate_agents7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max7)]
    all_answer7 = [[] for _ in range(N_max7)]
    for r in range(N_max7):
        for i, agent in enumerate(debate_agents7):
            inputs7 = [taskInfo, thinking6, answer6]
            if r > 0:
                inputs7 += all_thinking7[r-1] + all_answer7[r-1]
            thinking7_i, answer7_i = await agent(inputs7, debate7_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, substitute and simplify, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
            all_thinking7[r].append(thinking7_i)
            all_answer7[r].append(answer7_i)
    final_decision7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on simplified energy expression.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalize simplified energy, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    cot8_instruction = "Sub-task 8: Compare the simplified energy expression with the provided multiple-choice options and select the matching option exactly."
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent8([taskInfo, thinking7, answer7], cot8_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, select matching choice, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer