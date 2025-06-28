async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction1 = "Sub-task 1: Identify key experimental and analytical elements from the query about ChIP-seq on PFA-fixed versus PFA+DSG-fixed human B cells with IKAROS antibody and disappearing peaks and four candidate genomic region categories."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, extracting key elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: For each region category (active promoters and enhancers, repeats, introns of large genes, random genomic locations), define typical chromatin state, accessibility, and relevance in IKAROS ChIP-seq in B cells."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinking_mapping2 = {}
    answer_mapping2 = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, defining chromatin features, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinking_mapping2[answer2_i.content] = thinking2_i
        answer_mapping2[answer2_i.content] = answer2_i
    best_answer2 = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_mapping2[best_answer2]
    answer2 = answer_mapping2[best_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Analyze crosslinking properties of paraformaldehyde versus dual fixation with DSG+PFA and infer effects on direct versus indirect or artifactual ChIP signals."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, analyzing crosslinking, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    reflect_instruction4 = "Sub-task 4: Integrate crosslinking mechanisms and chromatin features for each region category to predict which peaks disappear when DSG is added."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent4(inputs4, reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, integrating predictions, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], "Critically evaluate the integration output for validity and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback on integration, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(inputs4, reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refining integration, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    debate_instruction5 = "Sub-task 5: Evaluate against known IKAROS biology and fixation artifacts which predicted region category best explains peaks lost in PFA+DSG fixation."
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                prev = all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4] + prev, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating category, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on which region category best explains disappearing peaks.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent5.id}, deciding category, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer