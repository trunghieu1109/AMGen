async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Characterize the two fixation chemistries PFA alone versus PFA+DSG and how each crosslinker captures protein–DNA and protein–protein interactions in a ChIP-seq experiment, given that peaks vanish when DSG is added."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, characterizing fixation chemistries, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Based on Sub-task 1 outputs, determine the impact of adding DSG on detection of direct versus indirect transcription factor DNA binding, considering retention of IKAROS–DNA interactions under PFA and loss under PFA+DSG due to extensive protein crosslinking."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, assessing DSG impact, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinking_map[answer2_i.content] = thinking2_i
        answer_map[answer2_i.content] = answer2_i
    most_common = Counter(possible_answers).most_common(1)[0][0]
    thinking2, answer2 = thinking_map[most_common], answer_map[most_common]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_reflect_instruction = "Sub-task 3: Survey which genomic features are known to host indirect, protein-mediated transcription factor binding that would be lost under stronger protein–protein crosslinking constraints. Options: active promoters/enhancers, repetitive elements, introns of large genes, or random regions."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, surveying genomic features, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent([taskInfo, thinking3, answer3], "Please review the surveyed genomic features for coverage and logical consistency, and indicate if valid.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined survey, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    debate_instruction = "Sub-task 4: Integrate the insights to rank genomic features by their likelihood to display disappearing IKAROS peaks under PFA+DSG fixation and select the top candidate."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking = [[] for _ in range(self.max_round)]
    all_answer = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking[r-1] + all_answer[r-1]
                thinking4, answer4 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking[r].append(thinking4)
            all_answer[r].append(answer4)
    final_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 4: Make final decision on the top genomic feature.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer