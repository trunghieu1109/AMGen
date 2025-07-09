async def forward_195(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Compute total energy E at maximum displacement by combining rest energy m c^2 and potential energy 1/2 k A^2.'
    cot_agent = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1,answer1 = await cot_agent([taskInfo],cot_instruction,is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing E, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response']={'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 2: Determine Lorentz factor gamma by solving gamma m c^2 = E and express gamma in terms of kA^2/(2 m c^2).'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_sc_instruction,'context':['user query','thinking of subtask 1','answer of subtask 1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking2_i,answer2_i = await cot_agents[i]([taskInfo,thinking1,answer1],cot_sc_instruction,is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, solving gamma, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content]=thinking2_i
        answermapping[answer2_i.content]=answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response']={'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_reflect_instruction = 'Sub-task 3: Derive v_max from gamma using gamma = 1/sqrt(1 - v^2/c^2) and simplify to explicit form in terms of kA^2/(2 m c^2).'
    cot_agent = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'],'Critic Agent',model=self.node_model,temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo,thinking1,answer1,thinking2,answer2]
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_reflect_instruction,'context':['user query','thinking of subtask 1','answer of subtask 1','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'Reflexion'}
    thinking3,answer3 = await cot_agent(cot_inputs,cot_reflect_instruction,0,is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, deriving v_max, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback,correct = await critic_agent([taskInfo,thinking3,answer3],'Review the derivation and provide limitations.',i,is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content=='True':
            break
        cot_inputs.extend([thinking3,answer3,feedback])
        thinking3,answer3 = await cot_agent(cot_inputs,cot_reflect_instruction,i+1,is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining v_max derivation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response']={'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    debate_instruction = 'Sub-task 4: Compare the derived v_max expression to the four choices and identify the matching option.'
    debate_agents = [LLMAgentBase(['thinking','answer'],'Debate Agent',model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':debate_instruction,'context':['user query','thinking of subtask 3','answer of subtask 3'],'agent_collaboration':'Debate'}
    for r in range(N_max_4):
        for i,agent in enumerate(debate_agents):
            if r==0:
                inputs = [taskInfo,thinking3,answer3]
            else:
                inputs = [taskInfo,thinking3,answer3] + all_thinking4[r-1] + all_answer4[r-1]
            thinking4_r,answer4_r = await agent(inputs,debate_instruction,r,is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_r.content}; answer: {answer4_r.content}")
            all_thinking4[r].append(thinking4_r)
            all_answer4[r].append(answer4_r)
    final_decision_agent = LLMAgentBase(['thinking','answer'],'Final Decision Agent',model=self.node_model,temperature=0.0)
    thinking4,answer4 = await final_decision_agent([taskInfo]+all_thinking4[-1]+all_answer4[-1],'Sub-task 4: Make final selection of the matching choice.',is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response']={'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4,answer4,sub_tasks,agents)
    return final_answer, logs