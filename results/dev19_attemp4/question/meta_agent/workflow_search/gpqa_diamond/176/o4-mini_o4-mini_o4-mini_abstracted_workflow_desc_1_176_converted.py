async def forward_176(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = 'Sub-task 1: Extract and summarize given parameters: radius ratio, mass ratio, peak wavelength, radial velocities from the user query.'
    N = self.max_sc
    cot_agents1 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        'subtask_id': 'subtask_1',
        'instruction': cot_sc_instruction1,
        'context': ['user query'],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents1[i]([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, extracting parameters, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_agent1 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings1 + possible_answers1, 'Sub-task 1: Synthesize and select consistent parameter summary.', is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction2 = 'Sub-task 2: Identify the domain and governing physical principles: black-body radiation law, Wien displacement law, and Doppler shift considerations.'
    cot_agents2 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        'subtask_id': 'subtask_2',
        'instruction': cot_sc_instruction2,
        'context': ['user query', thinking1.content, answer1.content],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, identifying domain and principles, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_agent2 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, 'Sub-task 2: Synthesize and select consistent domain and principles.', is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst = 'Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.'
    cot_reflect_instruction3 = 'Sub-task 3: Derive the general expression for the luminosity ratio L1/L2 in terms of R and T using L ∝ R^2 T^4 and Wien displacement law. ' + reflect_inst
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        'subtask_id': 'subtask_3',
        'instruction': cot_reflect_instruction3,
        'context': ['user query', thinking1.content, answer1.content, thinking2.content, answer2.content],
        'agent_collaboration': 'Reflexion'
    }
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, deriving expression, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], 'Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly True in correct.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == 'True':
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining expression, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction4 = 'Sub-task 4: Compute the numerical luminosity ratio by substituting R1/R2 = 1.5 and T1 = T2 into the derived formula.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        'subtask_id': 'subtask_4',
        'instruction': cot_instruction4,
        'context': ['user query', thinking3.content, answer3.content],
        'agent_collaboration': 'CoT'
    }
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computing numerical ratio, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr5 = 'Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_instruction5 = 'Sub-task 5: Compare the computed luminosity ratio to the provided multiple-choice options and select the closest match. ' + debate_instr5
    debate_agents5 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {
        'subtask_id': 'subtask_5',
        'instruction': debate_instruction5,
        'context': ['user query', thinking4.content, answer4.content],
        'agent_collaboration': 'Debate'
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent5 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], 'Given all the above thinking and answers, reason over them carefully and provide a final answer.', is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs