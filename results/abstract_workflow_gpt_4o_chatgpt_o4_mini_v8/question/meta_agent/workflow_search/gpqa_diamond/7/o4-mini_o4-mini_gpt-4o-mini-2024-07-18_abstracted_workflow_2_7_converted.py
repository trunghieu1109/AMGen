async def forward_7(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot1_instruction = "Sub-task 1: Given candidate genes G1, G2, G3 and their mutant resistance results, extract and tabulate the resistance levels (% of control) for each single mutant (g1, g2, g3) and each double mutant (g1g2, g1g3, g2g3)."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, tabulating resistance levels, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    cot2_instruction = "Sub-task 2: Based on the resistance table from Sub-task 1, analyze how each single knockout affects resistance relative to wild-type and rank the single mutants by severity of resistance loss."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, analyzing single mutant severity, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    counter = Counter(possible_answers)
    best_answer = counter.most_common(1)[0][0]
    thinking2 = thinkingmapping[best_answer]
    answer2 = answermapping[best_answer]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    cot3_instruction = "Sub-task 3: Using the resistance table from Sub-task 1, analyze double mutant interactions relative to wild-type and their corresponding single mutants to identify additive, synergistic, or masking effects."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs3 = [taskInfo, thinking1, answer1]
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot3_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, analyzing double mutant interactions, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "Critically evaluate the double mutant interaction analysis for accuracy, completeness, and correct classification of interactions.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot3_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining interaction analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    cot4_instruction = "Sub-task 4: Given the severity ranking of single mutants (from Sub-task 2) and double mutant interactions (from Sub-task 3), infer which gene functions as the upstream transcription factor whose knockout abolishes resistance."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking2, answer2, thinking3, answer3], cot4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, inferring transcription factor identity, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    debate_instruction5 = "Sub-task 5: Based on single mutant severity (Sub-task 2) and double mutant interaction (Sub-task 3), determine the interaction between G1 and G3 by identifying which gene is epistatic and whether they exhibit redundancy or pleiotropy."
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking2, answer2, thinking3, answer3], debate_instruction5, r, is_sub_task=True)
            else:
                input_infos5 = [taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(input_infos5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining G1-G3 interaction, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on G1-G3 interaction type.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent5.id}, deciding G1-G3 interaction, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    cot6_instruction = "Sub-task 6: Map the inferred transcription factor (Sub-task 4) and G1-G3 interaction model (Sub-task 5) onto the provided answer choices and select the matching choice."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4, thinking5, answer5], cot6_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, mapping to answer choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer