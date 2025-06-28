async def forward_30(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot1_instruction = "Sub-task 1: Identify the major product of nitration of toluene with HNO3/H2SO4, considering directing effects of the methyl group."
    cot1_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot1_agent([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot1_agent.id}, identifying nitration product, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot2_sc_instruction = "Sub-task 2: Determine the product of benzylic oxidation of the major nitrotoluene isomer (para-nitrotoluene) with MnO2 in H2SO4."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    for agent in sc_agents:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot2_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, oxidizing benzylic methyl, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_map[answer2.content] = thinking2
        answer_map[answer2.content] = answer2
    most_common = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map[most_common]
    answer2 = answer_map[most_common]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    cot3_instruction = "Sub-task 3: Predict the crossed-aldol condensation product of p-nitrobenzaldehyde and acetone under aqueous NaOH, including carbon–carbon bond connectivity and E/Z geometry."
    cot3_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic3_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot3_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot3_agent(cot3_inputs, cot3_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot3_agent.id}, predicting aldol product, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic3_agent([taskInfo, thinking3, answer3], "Critically evaluate the predicted aldol product connectivity and geometry and point out any issues.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic3_agent.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot3_inputs.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot3_agent(cot3_inputs, cot3_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot3_agent.id}, refining aldol product, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot4_instruction = "Sub-task 4: Analyze the three-dimensional structure of the aldol condensation product to identify all symmetry elements present, including mirror planes, rotation axes, and inversion centers."
    cot4_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot4_agent([taskInfo, thinking3, answer3], cot4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot4_agent.id}, identifying symmetry elements, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    debate5_instruction = "Sub-task 5: Based on detected symmetry elements, assign the correct molecular point group for the aldol condensation product among C_s, C2h, C3, or D2h."
    debate5_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thoughts5 = [[] for _ in range(self.max_round)]
    all_answers5 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for agent in debate5_agents:
            inputs5 = [taskInfo, thinking4, answer4]
            if r > 0:
                inputs5.extend(all_thoughts5[r-1] + all_answers5[r-1])
            thinking5_r, answer5_r = await agent(inputs5, debate5_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assigning point group, thinking: {thinking5_r.content}; answer: {answer5_r.content}")
            all_thoughts5[r].append(thinking5_r)
            all_answers5[r].append(answer5_r)
    final_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_agent5([taskInfo] + all_thoughts5[-1] + all_answers5[-1], "Sub-task 5: Make final decision on molecular point group.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer