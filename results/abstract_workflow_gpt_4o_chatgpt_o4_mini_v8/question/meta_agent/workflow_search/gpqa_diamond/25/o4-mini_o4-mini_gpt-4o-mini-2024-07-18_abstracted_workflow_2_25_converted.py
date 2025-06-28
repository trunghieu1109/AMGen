async def forward_25(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction = 'Sub-task 1: Using information that Cyclohexene + A yields 8,8-diiodobicyclo[4.2.0]octan-7-one, identify structural features required in A, including functional groups, ring size, and unsaturation.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying A features, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 2: Based on features from Sub-task 1, list and describe the two candidate structures for A: 4,4-diiodocyclobut-2-en-1-one and 2,2-diiodoethen-1-one, highlighting functional groups and ring or chain length relative to requirements.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, describing candidates for A, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])
    cot_reflect_instruction = 'Sub-task 3: Based on Sub-tasks 1 and 2, analyze mechanistic plausibility of each candidate A forming the bicyclo[4.2.0]octan-7-one with two iodines at C-8, and select the correct candidate.'
    cot_agent2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent2(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent2.id}, analyzing mechanism, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 'Critically evaluate mechanistic analysis for correctness and completeness and provide limitations.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, feedback on mechanism, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent2(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent2.id}, refining mechanism analysis, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])
    cot_instruction4 = 'Sub-task 4: Identify and explain key factors influencing reactivity of conjugated dienes, including conjugation, alkyl substitution, ring constraints, and stereochemistry, to establish evaluation criteria.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent3([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, identifying reactivity factors, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])
    debate_instruction = 'Sub-task 5: Using criteria from Sub-task 4, evaluate the reactivity of the dienes 2,3-dimethylbuta-1,3-diene; (2E,4E)-hexa-2,4-diene; cyclopenta-1,3-diene; (2Z,4Z)-hexa-2,4-diene, and determine their order from most to least reactive.'
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, evaluating dienes, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5_final, answer5_final = await final_decision_agent([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on reactivity ranking.', is_sub_task=True)
    agents.append(f'Final Decision agent, finalizing diene ranking, thinking: {thinking5_final.content}; answer: {answer5_final.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5_final.content}; answer - {answer5_final.content}')
    print('Step 5: ', sub_tasks[-1])
    cot_instruction6 = 'Sub-task 6: Compare the selected reactant A from Sub-task 3 and the diene ranking from Sub-task 5 against the provided answer choices, and select the matching option.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent4([taskInfo, thinking3, answer3, thinking5_final, answer5_final], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, selecting final choice, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer