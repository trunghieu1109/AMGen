async def forward_29(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = 'Sub-task 1: Formalize the model by introducing variables for the number of white/black colored rows (r_w, r_b) and columns (c_w, c_b), and describe how these determine which cells are occupied.'
    N = self.max_sc
    cot_sc_agents1 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_sc_instruction1, 'context': ['user query'], 'agent_collaboration': 'SC_CoT'}
    for agent in cot_sc_agents1:
        thinking, answer = await agent([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, formalizing model, thinking: {thinking.content}; answer: {answer.content}')
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_instr1 = 'Given all the above thinking and answers, find the most consistent and correct model formalization.'
    final_decision_agent1 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, 'Sub-task 1: Synthesize consistent model formalization. ' + final_instr1, is_sub_task=True)
    agents.append(f'Final Decision Agent {final_decision_agent1.id}, synthesizing model formalization, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction2 = 'Sub-task 2: Derive algebraic constraints from the monochromaticity conditions: each colored row/column is uniformly white or black, and at most one chip per cell implies initial bounds r_w+r_b ≤5 and c_w+c_b ≤5, along with an expression for total chips in terms of r_w,r_b,c_w,c_b.'
    cot_sc_agents2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction2, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for agent in cot_sc_agents2:
        thinking, answer = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f'CoT-SC agent {agent.id}, deriving constraints, thinking: {thinking.content}; answer: {answer.content}')
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_instr2 = 'Given all the above thinking and answers, find the most consistent derivation of algebraic constraints.'
    final_decision_agent2 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, 'Sub-task 2: Synthesize consistent algebraic constraints. ' + final_instr2, is_sub_task=True)
    agents.append(f'Final Decision Agent {final_decision_agent2.id}, synthesizing constraints, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    debate_instr = 'Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_instruction3 = 'Sub-task 3: Incorporate the maximality requirement to show that any empty cell must be blocked by a conflicting row or column color, leading to the key constraint that either all rows are colored (r_w+r_b=5) or all columns are colored (c_w+c_b=5).' + debate_instr
    debate_agents3 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking3 = [[] for _ in range(N_max)]
    all_answer3 = [[] for _ in range(N_max)]
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': debate_instruction3, 'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2'], 'agent_collaboration': 'Debate'}
    for r in range(N_max):
        for agent in debate_agents3:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1], debate_instruction3, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, considering maximality constraint, thinking: {thinking.content}; answer: {answer.content}')
            all_thinking3[r].append(thinking)
            all_answer3[r].append(answer)
    final_instr3 = 'Given all the above thinking and answers, reason over them carefully and provide a final answer.'
    final_decision_agent3 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], 'Sub-task 3: Synthesize debate outcomes and finalize maximality constraint demonstration. ' + final_instr3, is_sub_task=True)
    agents.append(f'Final Decision Agent {final_decision_agent3.id}, synthesizing maximality constraint, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_instruction4 = 'Sub-task 4: Solve the resulting system to enumerate all valid (r_w,r_b,c_w,c_b) assignments satisfying the constraints, and sum over these cases to obtain the final count of maximal configurations.'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, enumerating valid assignments and counting configurations, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_instruction4, 'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3'], 'agent_collaboration': 'CoT', 'response': {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs