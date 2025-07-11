async def forward_191(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Parse the problem statement and extract R, r, s, q, l, L, θ and their relationships"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing problem statement, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response']={'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Identify relevant physics principles and boundary conditions for uncharged conducting sphere with off-center cavity containing charge +q"
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_sc_instruction,'context':['user query','thinking1','answer1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identifying physics principles, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinking_mapping[answer2_i.content] = thinking2_i
        answer_mapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2_content]
    answer2 = answer_mapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response']={'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Derive the electric field magnitude at point P outside the conductor using net enclosed charge = +q"
    cot_agent3 = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_instruction3,'context':['user query','thinking2','answer2'],'agent_collaboration':'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, deriving E formula, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response']={'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Compare derived E expression with the four choices and select the matching one"
    cot_agent4 = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':cot_instruction4,'context':['user query','thinking3','answer3'],'agent_collaboration':'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, selecting matching choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response']={'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs