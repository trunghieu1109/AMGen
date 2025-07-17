async def forward_155(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Determine the stereochemical outcome of mCPBA epoxidation for (E)-oct-4-ene and (Z)-oct-4-ene, identifying which products are enantiomeric pairs and which are meso compound.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing epoxidation stereochemistry, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction2 = 'Sub-task 2: Based on the output from Sub-task 1, enumerate all distinct stereoisomers present in the combined reaction mixture and classify their relationships as enantiomers or meso.'
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    possible_thinkings2 = []
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction2, 'context': ['user query', 'thinking1', 'answer1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents2[i].id}, enumerating stereoisomers, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_instr2 = 'Given all the above thinking and answers, select the most consistent enumeration of stereoisomers and their classifications.'
    final_decision_agent2 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, 'Sub-task 2: Synthesize and choose the most consistent answer for stereoisomer enumeration. ' + final_instr2, is_sub_task=True)
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    cot_sc_instruction3 = 'Sub-task 3: Predict the number of peaks observed on an achiral reverse-phase HPLC column for the mixture of epoxide stereoisomers, given enantiomers coelute but diastereomers are resolved.'
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers3 = []
    possible_thinkings3 = []
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_sc_instruction3, 'context': ['user query', 'thinking2', 'answer2'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents3[i].id}, predicting achiral HPLC peaks, thinking: {thinking3_i.content}; answer: {answer3_i.content}')
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_instr3 = 'Given all the above thinking and answers, select the most consistent prediction for number of achiral HPLC peaks.'
    final_decision_agent3 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, 'Sub-task 3: Synthesize prediction for achiral HPLC. ' + final_instr3, is_sub_task=True)
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_sc_instruction4 = 'Sub-task 4: Predict the number of peaks observed on a chiral HPLC column for the mixture of epoxide stereoisomers, given full baseline resolution of enantiomers and meso compound.'
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers4 = []
    possible_thinkings4 = []
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_sc_instruction4, 'context': ['user query', 'thinking2', 'answer2'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking2, answer2], cot_sc_instruction4, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents4[i].id}, predicting chiral HPLC peaks, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_instr4 = 'Given all the above thinking and answers, select the most consistent prediction for number of chiral HPLC peaks.'
    final_decision_agent4 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking2, answer2] + possible_thinkings4 + possible_answers4, 'Sub-task 4: Synthesize prediction for chiral HPLC. ' + final_instr4, is_sub_task=True)
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    debate_instr = 'Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_instruction5 = 'Sub-task 5: Select the correct answer choice for the numbers of peaks on the standard and chiral HPLC columns based on the predictions from Sub-task 3 and Sub-task 4. ' + debate_instr
    debate_agents5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N5 = self.max_round
    all_thinking5 = [[] for _ in range(N5)]
    all_answer5 = [[] for _ in range(N5)]
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': debate_instruction5, 'context': ['user query', 'thinking3', 'answer3', 'thinking4', 'answer4'], 'agent_collaboration': 'Debate'}
    for r in range(N5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                input_infos5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(input_infos5, debate_instruction5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_instr5 = 'Given all the above thinking and answers, reason over them carefully and provide a final answer.'
    final_decision_agent5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Select the final answer choice. ' + final_instr5, is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs