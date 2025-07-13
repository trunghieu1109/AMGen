async def forward_196(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction = 'Sub-task 1: Extract key functional groups and structural fragments from IR and 1H NMR data, identifying possible functional groups such as COOH, aromatic para-disubstitution, and alkyl chain'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    final_instr = 'Synthesize and choose the most consistent functional assignments from Sub-task 1'
    thinking1, answer1 = await final_decision_agent([taskInfo] + possible_thinkings + possible_answers, 'Sub-task 1: ' + final_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({'subtask_id':'subtask_1','instruction':cot_sc_instruction,'agent_collaboration':'SC_CoT','response':{'thinking':thinking1,'answer':answer1})
    print('Step 1:', sub_tasks[-1])
    debate_instr = 'Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_instruction = 'Sub-task 2: Assign substitution pattern and confirm alkyl side chain based on Sub-task 1 output.' + debate_instr
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision2 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], 'Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.', is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({'subtask_id':'subtask_2','instruction':debate_instruction,'agent_collaboration':'Debate','response':{'thinking':thinking2,'answer':answer2})
    print('Step 2:', sub_tasks[-1])
    cot_sc_instruction3 = 'Sub-task 3: Predict the effect of red phosphorus and HI on the identified acid, determining decarboxylation or reduction pathway and core skeleton of product.'
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    for agent in cot_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision3 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings3 + possible_answers3, 'Sub-task 3: Synthesize reaction outcome and product skeleton. Given all above, provide a final decision.', is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({'subtask_id':'subtask_3','instruction':cot_sc_instruction3,'agent_collaboration':'SC_CoT','response':{'thinking':thinking3,'answer':answer3})
    print('Step 3:', sub_tasks[-1])
    debate_instruction4 = 'Sub-task 4: Select final product structure from choices ensuring consistency with para-disubstituted aromatic core, sec-butyl chain, and decarboxylation outcome.' + debate_instr
    debate_agents4 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for agent in debate_agents4:
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking2, answer2, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                inputs4 = [taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(inputs4, debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision4 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision4([taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], 'Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.', is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append({'subtask_id':'subtask_4','instruction':debate_instruction4,'agent_collaboration':'Debate','response':{'thinking':thinking4,'answer':answer4})
    print('Step 4:', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs