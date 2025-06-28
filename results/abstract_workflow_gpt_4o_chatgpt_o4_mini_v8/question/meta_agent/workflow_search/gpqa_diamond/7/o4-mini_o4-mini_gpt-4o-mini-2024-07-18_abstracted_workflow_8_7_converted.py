async def forward_7(self, taskInfo):
    from collections import Counter
    print('Task Requirement:', taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1
    cot_instruction = 'Sub-task 1: Extract resistance percentages for g1, g2, g3, g1g2, g1g3, g2g3 from the experiment data.'
    cot_agent1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, extracting resistances, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print('Step 1:', sub_tasks[-1])

    # Sub-task 2
    cot_sc_instruction = 'Sub-task 2: Compute susceptibility (100 minus resistance) per genotype using Sub-task 1 output.'
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking', 'answer'], 'Self-Consistency CoT Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, computing susceptibility, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinking_map2[answer2_i.content] = thinking2_i
        answer_map2[answer2_i.content] = answer2_i
    most_common2 = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[most_common2]
    answer2 = answer_map2[most_common2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print('Step 2:', sub_tasks[-1])

    # Sub-task 3
    cot_reflect_instruction = 'Sub-task 3: Rank genotypes by susceptibility severity from highest to lowest.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, ranking genotypes, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], 'Evaluate the ranking for correctness and consistency.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == 'True':
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining ranking, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print('Step 3:', sub_tasks[-1])

    # Stage 2
    # Sub-task 4
    cot_instruction4 = 'Sub-task 4: Identify which single knockout has 100% susceptibility indicating a transcription factor.'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, identifying transcription factor, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print('Step 4:', sub_tasks[-1])

    # Sub-task 5
    debate_instruction5 = 'Sub-task 5: Analyze double-mutant phenotypes relative to singles (g1 vs g3, g1 vs g2, g2 vs g3) to infer redundancy or epistasis.'
    debate_agents5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing interactions, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Make final inference on gene-gene interactions.', is_sub_task=True)
    agents.append(f"Final Decision agent, deciding interactions, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print('Step 5:', sub_tasks[-1])

    # Sub-task 6
    cot_instruction6 = 'Sub-task 6: Combine transcription factor and interaction inference to select the correct multiple-choice conclusion.'
    cot_agent6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, selecting conclusion, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print('Step 6:', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer