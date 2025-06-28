async def forward_6(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot1([taskInfo], "Sub-task 1: Extract the physical reaction gamma+gamma->e+e- and given CMB photon energy.", is_sub_task=True)
    agents.append(f"CoT agent {cot1.id}, extracting reaction and epsilon_CMB, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    cot2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot2([taskInfo, thinking1, answer1], "Sub-task 2: Identify and classify key physics principle E_gamma * epsilon_CMB >= (m_e c^2)^2.", is_sub_task=True)
    agents.append(f"CoT agent {cot2.id}, classifying threshold principle, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    cot3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot3([taskInfo], "Sub-task 3: Recall electron rest mass energy m_e c^2 = 511 keV and convert to eV.", is_sub_task=True)
    agents.append(f"CoT agent {cot3.id}, recalling and converting m_e c^2, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "SC-CoT Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible4 = []
    thinking_map4 = {}
    answer_map4 = {}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking2, answer2], "Sub-task 4: Formulate algebraic expression E_gamma(threshold) = (m_e c^2)^2 / epsilon_CMB.", is_sub_task=True)
        agents.append(f"SC-CoT agent {cot_agents4[i].id}, formulating threshold expression, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible4.append(answer4_i.content)
        thinking_map4[answer4_i.content] = thinking4_i
        answer_map4[answer4_i.content] = answer4_i
    counter4 = Counter(possible4)
    consensus4 = counter4.most_common(1)[0][0]
    thinking4 = thinking_map4[consensus4]
    answer4 = answer_map4[consensus4]
    agents.append(f"Consensus SC-CoT for Sub-task 4: answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    cot5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot5([taskInfo, thinking3, answer3], "Sub-task 5: Compute the squared electron rest mass energy (5.11×10^5 eV)^2.", is_sub_task=True)
    agents.append(f"CoT agent {cot5.id}, computing square of rest mass energy, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    debate_agents6 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max6)]
    all_answer6 = [[] for _ in range(N_max6)]
    for r in range(N_max6):
        for agent in debate_agents6:
            if r == 0:
                inputs6 = [taskInfo, thinking5, answer5]
            else:
                inputs6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
            thinking6_i, answer6_i = await agent(inputs6, "Sub-task 6: Divide squared energy by epsilon_CMB and convert result into GeV.", is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing threshold energy, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
            all_thinking6[r].append(thinking6_i)
            all_answer6[r].append(answer6_i)
    final6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on threshold energy value in GeV.", is_sub_task=True)
    agents.append(f"Final Decision agent {final6.id}, selecting threshold energy, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    cot7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic7 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs7 = [taskInfo, thinking6, answer6]
    thinking7, answer7 = await cot7(inputs7, "Sub-task 7: Compare computed threshold with answer choices and select the matching option.", is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot7.id}, initial selection, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(self.max_round):
        feedback7, correct7 = await critic7([taskInfo, thinking7, answer7], "Review the selection against choices and provide limitations.", is_sub_task=True)
        agents.append(f"Critic agent {critic7.id}, feedback on selection, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        inputs7 += [thinking7, answer7, feedback7]
        thinking7, answer7 = await cot7(inputs7, "Sub-task 7: Refine selection of matching choice.", is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot7.id}, refining selection, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer