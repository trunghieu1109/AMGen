async def forward_159(self, taskInfo):
    from collections import Counter
    print('Task Requirement:', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Show that as N approaches infinity, a regular N-sided polygon with apothem a becomes a circular aperture of radius a.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing polygon limit, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response']={'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1:', sub_tasks[-1])
    debate_instruction = 'Sub-task 2: Derive the angular positions of the first two diffraction minima for a circular aperture of radius a using Fraunhofer diffraction and the first zero formula with small angle approximation. Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_agents = [LLMAgentBase(['thinking','answer'],'Debate Agent',model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':debate_instruction,'context':['user query','thinking of subtask_1','answer of subtask_1'],'agent_collaboration':'Debate'}
    for r in range(N_max):
        for agent in debate_agents:
            if r==0:
                thinking2, answer2 = await agent([taskInfo,thinking1,answer1], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo,thinking1,answer1] + all_thinking[r-1] + all_answer[r-1]
                thinking2, answer2 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking[r].append(thinking2)
            all_answer[r].append(answer2)
    final_instr = 'Given all the above thinking and answers, reason over them carefully and provide a final answer.'
    final_agent2 = LLMAgentBase(['thinking','answer'],'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking2_final, answer2_final = await final_agent2([taskInfo,thinking1,answer1] + all_thinking[-1] + all_answer[-1], 'Sub-task 2:'+final_instr, is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent2.id}, thinking: {thinking2_final.content}; answer: {answer2_final.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2['response']={'thinking':thinking2_final,'answer':answer2_final}
    logs.append(subtask_desc2)
    print('Step 2:', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 3: Convert the derived expression into the form λ/a and compare with the given answer choices to select the correct one.'
    N_sc = self.max_sc
    sc_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_sc_instruction,'context':['user query','thinking of subtask_2','answer of subtask_2'],'agent_collaboration':'SC_CoT'}
    for agent in sc_agents:
        thinking3, answer3 = await agent([taskInfo, thinking2_final, answer2_final], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings.append(thinking3)
        possible_answers.append(answer3)
    final_agent3 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking3_final, answer3_final = await final_agent3([taskInfo, thinking2_final, answer2_final] + possible_thinkings + possible_answers, 'Sub-task 3: Synthesize and choose the most consistent answer.', is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent3.id}, thinking: {thinking3_final.content}; answer: {answer3_final.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    subtask_desc3['response']={'thinking':thinking3_final,'answer':answer3_final}
    logs.append(subtask_desc3)
    print('Step 3:', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3_final, answer3_final, sub_tasks, agents)
    return final_answer, logs