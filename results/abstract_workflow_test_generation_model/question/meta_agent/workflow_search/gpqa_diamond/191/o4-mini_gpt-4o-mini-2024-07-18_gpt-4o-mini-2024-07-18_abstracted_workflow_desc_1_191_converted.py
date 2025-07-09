async def forward_191(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks=[]
    agents=[]
    logs=[]

    cot_instruction='Sub-task 1: Extract and classify all given geometrical and physical parameters (R, r, s, q, positions of cavity center and point P, angles θ, distances l and L) from the query.'
    cot_agent=LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1={'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1,answer1=await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response']={'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    instruction2='Sub-task 2: Identify and articulate the boundary conditions and conductor properties: uncharged conducting shell, cavity with charge, induced surface charges, equipotential condition.'
    N=self.max_sc
    cot_agents2=[LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers=[]
    thinking_mapping={}
    answer_mapping={}
    subtask_desc2={'subtask_id':'subtask_2','instruction':instruction2,'context':['user query','thinking of subtask 1','answer of subtask 1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        t_i,a_i=await cot_agents2[i]([taskInfo,thinking1,answer1], instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, articulating boundary conditions, thinking: {t_i.content}; answer: {a_i.content}")
        possible_answers.append(a_i.content)
        thinking_mapping[a_i.content]=t_i
        answer_mapping[a_i.content]=a_i
    answer2_content=Counter(possible_answers).most_common(1)[0][0]
    thinking2=thinking_mapping[answer2_content]
    answer2=answer_mapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response']={'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    instruction3='Sub-task 3: Apply electrostatic theorems to determine induced charge distribution on inner and outer surfaces and compute net external charge.'
    N3=self.max_sc
    cot_agents3=[LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers3=[]
    thinking_mapping3={}
    answer_mapping3={}
    subtask_desc3={'subtask_id':'subtask_3','instruction':instruction3,'context':['user query','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'SC_CoT'}
    for i in range(N3):
        t_i,a_i=await cot_agents3[i]([taskInfo,thinking2,answer2], instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, computing induced charges, thinking: {t_i.content}; answer: {a_i.content}")
        possible_answers3.append(a_i.content)
        thinking_mapping3[a_i.content]=t_i
        answer_mapping3[a_i.content]=a_i
    answer3_content=Counter(possible_answers3).most_common(1)[0][0]
    thinking3=thinking_mapping3[answer3_content]
    answer3=answer_mapping3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response']={'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    instruction4='Sub-task 4: Use net external charge result and conductor equivalence to derive the expression for the electric field at point P (L > R).'  
    cot_agent4=LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent=LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N4=self.max_round
    subtask_desc4={'subtask_id':'subtask_4','instruction':instruction4,'context':['user query','thinking of subtask 3','answer of subtask 3'],'agent_collaboration':'Reflexion'}
    inputs4=[taskInfo,thinking3,answer3]
    thinking4,answer4=await cot_agent4(inputs4, instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, deriving E at P, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N4):
        feedback,correct=await critic_agent([taskInfo,thinking4,answer4], 'Please review the derived expression for the electric field at P and provide limitations.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content=='True':
            break
        inputs4.extend([thinking4,answer4,feedback])
        thinking4,answer4=await cot_agent4(inputs4, instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refining E derivation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response']={'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    instruction5='Sub-task 5: List the four provided multiple-choice expressions and compare each against the derived formula for E at P.'
    cot_agent5=LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc5={'subtask_id':'subtask_5','instruction':instruction5,'context':['user query','thinking of subtask 4','answer of subtask 4'],'agent_collaboration':'CoT'}
    thinking5,answer5=await cot_agent5([taskInfo,thinking4,answer4], instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, comparing choices, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response']={'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])

    instruction6='Sub-task 6: Select the choice (A, B, C, or D) whose formula matches the derived expression for the electric field at P.'
    debate_agents=[LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N6=self.max_round
    all_thinking6=[[] for _ in range(N6)]
    all_answer6=[[] for _ in range(N6)]
    subtask_desc6={'subtask_id':'subtask_6','instruction':instruction6,'context':['user query','thinking of subtask 5','answer of subtask 5'],'agent_collaboration':'Debate'}
    for r in range(N6):
        for i,agent in enumerate(debate_agents):
            if r==0:
                t_i,a_i=await agent([taskInfo,thinking5,answer5], instruction6, r, is_sub_task=True)
            else:
                t_i,a_i=await agent([taskInfo,thinking5,answer5]+all_thinking6[r-1]+all_answer6[r-1], instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t_i.content}; answer: {a_i.content}")
            all_thinking6[r].append(t_i)
            all_answer6[r].append(a_i)
    final_decision=LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6,answer6=await final_decision([taskInfo]+all_thinking6[-1]+all_answer6[-1], 'Sub-task 6: Make final decision on the correct choice.', is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response']={'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])

    final_answer=await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs