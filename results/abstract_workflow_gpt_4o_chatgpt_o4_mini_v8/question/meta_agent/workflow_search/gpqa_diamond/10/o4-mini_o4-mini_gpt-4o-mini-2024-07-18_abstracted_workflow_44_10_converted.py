async def forward_10(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    cot1_instruction = 'Sub-task 1: Extract all key molecular-biology assertions from the question and the four answer choices, producing an itemized list of mechanisms and claims.'
    cot1_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot1_agent([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot1_agent.id}, extracting assertions, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])
    sc_instruction = 'Sub-task 2: Classify each extracted assertion by its functional category—translational frameshifting, proofreading exonuclease activity, apoptosis induction pathway, or pseudoknot structural dynamics.'
    N = self.max_sc
    sc_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_cats = []
    thinking_map = {}
    answer_map = {}
    for i in range(N):
        thinking2_i, answer2_i = await sc_agents[i]([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {sc_agents[i].id}, classifying assertion, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_cats.append(answer2_i.content)
        thinking_map[answer2_i.content] = thinking2_i
        answer_map[answer2_i.content] = answer2_i
    selected = Counter(possible_cats).most_common(1)[0][0]
    thinking2 = thinking_map[selected]
    answer2 = answer_map[selected]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])
    reflect_instr3 = 'Sub-task 3: Evaluate the validity of choice 1’s claim about SARS-CoV-2 programmed −1 ribosomal frameshifting and its conformational similarity to SARS-CoV frameshifting signals.'
    cot3_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic3 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot3_agent(inputs3, reflect_instr3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot3_agent.id}, evaluating choice1, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(self.max_round):
        feedback3, correct3 = await critic3([taskInfo, thinking3, answer3], 'Review the evaluation of choice1 and identify any inaccuracies.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic3.id}, feedback: {feedback3.content}; correct: {correct3.content}')
        if correct3.content == 'True':
            break
        inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot3_agent(inputs3, reflect_instr3, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot3_agent.id}, refining evaluation of choice1, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])
    cot4_instruction = 'Sub-task 4: Verify the accuracy of choice 2’s description of the SARS-CoV-2 nsp10/nsp14 ExoN heterodimer in mismatch repair and its purported role in preventing dsRNA breakdown.'
    cot4_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot4_agent([taskInfo, thinking2, answer2], cot4_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot4_agent.id}, evaluating choice2, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])
    cot5_instruction = 'Sub-task 5: Assess choice 3’s assertion that SARS-CoV-2 ORF3a exclusively triggers extrinsic apoptosis via caspase-8 without altering Bcl-2 levels.'
    cot5_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot5_agent([taskInfo, thinking2, answer2], cot5_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot5_agent.id}, evaluating choice3, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5: ', sub_tasks[-1])
    cot6_instruction = 'Sub-task 6: Examine choice 4’s proposition that the in vitro frameshifting rate scales linearly with the number of pseudoknot conformations and that both SARS-CoV and SARS-CoV-2 frameshift signals display exactly two tension-induced folds.'
    cot6_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot6_agent([taskInfo, thinking2, answer2], cot6_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot6_agent.id}, evaluating choice4, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ', sub_tasks[-1])
    debate7_instruction = 'Sub-task 7: Integrate the findings from subtasks 3–6 to identify which one of the four statements is incorrect.'
    debate_agents7 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = [[] for _ in range(self.max_round)]
    all_answer7 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents7):
            inputs7 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6]
            if r > 0:
                inputs7 += all_thinking7[r-1] + all_answer7[r-1]
            thinking7_r, answer7_r = await agent(inputs7, debate7_instruction, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, thinking: {thinking7_r.content}; answer: {answer7_r.content}')
            all_thinking7[r].append(thinking7_r)
            all_answer7[r].append(answer7_r)
    final_decision7 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision7([taskInfo] + all_thinking7[-1] + all_answer7[-1], 'Sub-task 7: Decide which statement is incorrect.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision7.id}, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Step 7: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer