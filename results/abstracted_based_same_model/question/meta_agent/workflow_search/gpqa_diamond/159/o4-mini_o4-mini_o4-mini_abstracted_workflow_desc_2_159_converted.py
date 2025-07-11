async def forward_159(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks=[]
    agents=[]
    logs=[]

    cot_instruction1='Sub-task 1: Recognize that as N approaches infinity, the N-sided polygon aperture approaches a circular aperture perpendicular to the z-axis.'
    cot_agent1=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc1={'subtask_id':'subtask_1','instruction':cot_instruction1,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1,answer1=await cot_agent1([taskInfo],cot_instruction1,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, recognizing aperture shape limit, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response']={'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ',sub_tasks[-1])

    N=self.max_sc
    cot_agents2=[LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    cot_sc_instruction2='Sub-task 2: Determine relation between apothem a and circular aperture parameters, identifying diameter D=2a.'
    subtask_desc2={'subtask_id':'subtask_2','instruction':cot_sc_instruction2,'context':['user query',thinking1,answer1],'agent_collaboration':'SC_CoT'}
    possible_answers2=[]
    thinking_mapping2={}
    answer_mapping2={}
    for agent2 in cot_agents2:
        thinking2,answer2=await agent2([taskInfo,thinking1,answer1],cot_sc_instruction2,is_sub_task=True)
        agents.append(f'CoT-SC agent {agent2.id}, relating a to D, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers2.append(answer2.content)
        thinking_mapping2[answer2.content]=thinking2
        answer_mapping2[answer2.content]=answer2
    answer2_content=Counter(possible_answers2).most_common(1)[0][0]
    thinking2=thinking_mapping2[answer2_content]
    answer2=answer_mapping2[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response']={'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ',sub_tasks[-1])

    cot_instruction3='Sub-task 3: Recall the first two zeros of J1: j_{1,1} ≈ 3.8317 and j_{1,2} ≈ 7.0156 for Airy pattern minima.'
    cot_agent3=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc3={'subtask_id':'subtask_3','instruction':cot_instruction3,'context':['user query',thinking2,answer2],'agent_collaboration':'CoT'}
    thinking3,answer3=await cot_agent3([taskInfo,thinking2,answer2],cot_instruction3,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, recalling Bessel zeros, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response']={'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ',sub_tasks[-1])

    cot_instruction4='Sub-task 4: Write general small-angle formula for nth minimum θ_n = j_{1,n} * λ / (π * D).'
    cot_agent4=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc4={'subtask_id':'subtask_4','instruction':cot_instruction4,'context':['user query',thinking3,answer3],'agent_collaboration':'CoT'}
    thinking4,answer4=await cot_agent4([taskInfo,thinking3,answer3],cot_instruction4,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, formulating θ_n formula, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response']={'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ',sub_tasks[-1])

    cot_instruction5='Sub-task 5: Compute θ_1 = j_{1,1} * λ / (π * D) and express in terms of a using D=2a.'
    cot_agent5=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc5={'subtask_id':'subtask_5','instruction':cot_instruction5,'context':['user query',thinking4,answer4],'agent_collaboration':'CoT'}
    thinking5,answer5=await cot_agent5([taskInfo,thinking4,answer4],cot_instruction5,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent5.id}, computing θ1, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response']={'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ',sub_tasks[-1])

    cot_instruction6='Sub-task 6: Compute θ_2 = j_{1,2} * λ / (π * D) and express in terms of a using D=2a.'
    cot_agent6=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc6={'subtask_id':'subtask_6','instruction':cot_instruction6,'context':['user query',thinking4,answer4],'agent_collaboration':'CoT'}
    thinking6,answer6=await cot_agent6([taskInfo,thinking4,answer4],cot_instruction6,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, computing θ2, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response']={'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    print('Step 6: ',sub_tasks[-1])

    cot_reflect_instruction7='Sub-task 7: Calculate Δθ = θ_2 - θ_1 and also compute symmetric separation 2θ_1 to confirm interpretation.'
    cot_agent7=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    critic_agent7=LLMAgentBase(['feedback','correct'],'Critic Agent',model=self.node_model,temperature=0.0)
    subtask_desc7={'subtask_id':'subtask_7','instruction':cot_reflect_instruction7,'context':['user query',thinking5,answer5,thinking6,answer6],'agent_collaboration':'Reflexion'}
    inputs7=[taskInfo,thinking5,answer5,thinking6,answer6]
    thinking7,answer7=await cot_agent7(inputs7,cot_reflect_instruction7,0,is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent7.id}, initial Δθ and symmetry calc, thinking: {thinking7.content}; answer: {answer7.content}')
    for i in range(self.max_round):
        feedback7,correct7=await critic_agent7([taskInfo,thinking7,answer7],'Review the calculated separations Δθ and 2θ1 for interpretation correctness.',i,is_sub_task=True)
        agents.append(f'Critic agent {critic_agent7.id}, feedback on Δθ calc, thinking: {feedback7.content}; answer: {correct7.content}')
        if correct7.content=='True':
            break
        inputs7.extend([thinking7,answer7,feedback7])
        thinking7,answer7=await cot_agent7(inputs7,cot_reflect_instruction7,i+1,is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent7.id}, refined Δθ calc, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response']={'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)
    print('Step 7: ',sub_tasks[-1])

    debate_instruction8='Sub-task 8: Compare Δθ to the choices and select the matching option letter.'
    debate_agents8=[LLMAgentBase(['thinking','answer'],'Debate Agent',model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    all_thinking8=[[] for _ in range(self.max_round)]
    all_answer8=[[] for _ in range(self.max_round)]
    subtask_desc8={'subtask_id':'subtask_8','instruction':debate_instruction8,'context':['user query',thinking7,answer7],'agent_collaboration':'Debate'}
    for r in range(self.max_round):
        for agent8 in debate_agents8:
            if r==0:
                thinking8,answer8=await agent8([taskInfo,thinking7,answer7],debate_instruction8,r,is_sub_task=True)
            else:
                thinking8,answer8=await agent8([taskInfo,thinking7,answer7]+all_thinking8[r-1]+all_answer8[r-1],debate_instruction8,r,is_sub_task=True)
            agents.append(f'Debate agent {agent8.id}, round {r}, comparing Δθ to choices, thinking: {thinking8.content}; answer: {answer8.content}')
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision_agent8=LLMAgentBase(['thinking','answer'],'Final Decision Agent',model=self.node_model,temperature=0.0)
    thinking8,answer8=await final_decision_agent8([taskInfo]+all_thinking8[-1]+all_answer8[-1],'Sub-task 8: Make final decision on the choice letter.',is_sub_task=True)
    agents.append(f'Final Decision agent, selecting choice, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    subtask_desc8['response']={'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)
    print('Step 8: ',sub_tasks[-1])

    final_answer=await self.make_final_answer(thinking8,answer8,sub_tasks,agents)
    return final_answer,logs