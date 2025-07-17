async def forward_170(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs =  []

    cot_sc_instruction = 'Sub-task 1: Identify and classify each substituent (–CH3, –COOC2H5, –Cl, –NO2, –C2H5, –COOH) as activating or deactivating and as ortho/para or meta directors, and record their Hammett σ constants.'
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinking1 = []
    possible_answer1 = []
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_sc_instruction,'context':['user query'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents_1:
        thinking1_i, answer1_i = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}')
        possible_thinking1.append(thinking1_i)
        possible_answer1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinking1 + possible_answer1, 'Sub-task 1: Synthesize the most consistent classification of substituents based on the above thoughts.', is_sub_task=True)
    agents.append(f'Final Decision Agent, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    debate_instruction2 = 'Sub-task 2: Collect quantitative data for each substrate: list M0 (substrate molar mass), Mp (para‐bromo product molar mass), and literature-based para:ortho:meta percentages. Construct a table and explicitly state the weight_fraction formula: weight_fraction_i = (xₚ_i × Mₚ_i) / Σ_j(xₚ_j × Mₚ_j). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_agents2 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    M_max = self.max_round
    all_thinking2 = [[] for _ in range(M_max)]
    all_answer2 = [[] for _ in range(M_max)]
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':debate_instruction2,'context':['user query','response of subtask_1'],'agent_collaboration':'Debate'}
    for r in range(M_max):
        for agent in debate_agents2:
            if r==0:
                thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], debate_instruction2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2_i, answer2_i = await agent(inputs, debate_instruction2, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
            all_thinking2[r].append(thinking2_i)
            all_answer2[r].append(answer2_i)
    final_decision_agent2 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], 'Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.', is_sub_task=True)
    agents.append(f'Final Decision Agent, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    cot_sc_instruction3 = 'Sub-task 3: Compute for each substrate the numerator (xₚ × Mₚ) and then the normalized weight fraction using the formula from Sub-task 2. Produce a list of weight fractions with units and note any assumptions made.'
    cot_agents_3 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinking3 = []
    possible_answer3 = []
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_sc_instruction3,'context':['user query','thinking of subtask_2','answer of subtask_2'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents_3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}')
        possible_thinking3.append(thinking3_i)
        possible_answer3.append(answer3_i)
    final_decision_agent3 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinking3 + possible_answer3, 'Sub-task 3: Synthesize the most consistent weight fractions based on the above thoughts.', is_sub_task=True)
    agents.append(f'Final Decision Agent, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    debate_instruction4 = 'Sub-task 4: Arrange the six substrates in ascending order of the computed weight fractions. Engage in a debate to challenge and justify each adjacent ordering, focusing on cases like –COOH vs. –COOC2H5 and –Cl vs. alkyls. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_agents4 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(M_max)]
    all_answer4 = [[] for _ in range(M_max)]
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':debate_instruction4,'context':['user query','thinking of subtask_3','answer of subtask_3'],'agent_collaboration':'Debate'}
    for r in range(M_max):
        for agent in debate_agents4:
            if r==0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(inputs, debate_instruction4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], 'Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.', is_sub_task=True)
    agents.append(f'Final Decision Agent, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    reflect_inst = 'Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.'
    cot_reflect_instruction5 = 'Sub-task 5: Perform a sensitivity analysis on the closest weight-fraction pairs by slightly varying xₚ or Mₚ within realistic experimental margins to confirm that the ranking is robust.' + reflect_inst
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':cot_reflect_instruction5,'context':['user query','thinking of subtask_4','answer of subtask_4'],'agent_collaboration':'Reflexion'}
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], 'Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly True in correct', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}')
        if correct5.content == 'True':
            break
        cot_inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    cot_instruction6 = 'Sub-task 6: Map the final ascending order from Sub-task 4 and the sensitivity-confirmed order from Sub-task 5 to the provided answer choices (choice1–choice4) and select the matching option, explaining how it corresponds.'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':cot_instruction6,'context':['user query','thinking of subtask_4','answer of subtask_4','thinking of subtask_5','answer of subtask_5'],'agent_collaboration':'CoT'}
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs