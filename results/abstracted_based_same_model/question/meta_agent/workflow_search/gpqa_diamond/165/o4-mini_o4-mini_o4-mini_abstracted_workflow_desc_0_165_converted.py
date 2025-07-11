async def forward_165(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks=[]
    agents=[]
    logs=[]

    cot_instruction='Sub-task 1: Parse the given Lagrangian to extract fields N_R, φ, S, H, their gauge charges, couplings y_i, g_{iα}, and vacuum expectation values ⟨φ⟩=x, ⟨h⟩=v.'
    cot_agent=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc1={'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1,answer1=await cot_agent([taskInfo],cot_instruction,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, parsing Lagrangian, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ',sub_tasks[-1])
    subtask_desc1['response']={'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction='Sub-task 2: Identify the global symmetry spontaneously broken by the VEVs and determine which combination corresponds to the would-be Goldstone boson H_2.'
    N=self.max_sc
    cot_agents=[LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers=[]
    thinkingmapping={}
    answermapping={}
    subtask_desc2={'subtask_id':'subtask_2','instruction':cot_sc_instruction,'context':['user query','subtask_1 thinking','subtask_1 answer'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking2_i,answer2_i=await cot_agents[i]([taskInfo,thinking1,answer1],cot_sc_instruction,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, identifying broken symmetry, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content]=thinking2_i
        answermapping[answer2_i.content]=answer2_i
    answer2_content=Counter(possible_answers).most_common(1)[0][0]
    thinking2=thinkingmapping[answer2_content]
    answer2=answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ',sub_tasks[-1])
    subtask_desc2['response']={'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)

    cot_reflect_instruction='Sub-task 3: Explain why H_2 is massless at tree level and describe the necessity of radiative corrections to generate its mass.'
    cot_agent_ref=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    critic_agent=LLMAgentBase(['feedback','correct'],'Critic Agent',model=self.node_model,temperature=0.0)
    N_max=self.max_round
    cot_inputs=[taskInfo,thinking1,answer1,thinking2,answer2]
    subtask_desc3={'subtask_id':'subtask_3','instruction':cot_reflect_instruction,'context':['user query','subtask_1 thinking','subtask_1 answer','subtask_2 thinking','subtask_2 answer'],'agent_collaboration':'Reflexion'}
    thinking3,answer3=await cot_agent_ref(cot_inputs,cot_reflect_instruction,0,is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_ref.id}, explaining masslessness, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback,correct=await critic_agent([taskInfo,thinking3,answer3],'Please review the explanation and provide its limitations.',i,is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, review round {i}, feedback: {feedback.content}; correct: {correct.content}')
        if correct.content=='True':
            break
        cot_inputs.extend([thinking3,answer3,feedback])
        thinking3,answer3=await cot_agent_ref(cot_inputs,cot_reflect_instruction,i+1,is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_ref.id}, refining explanation, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ',sub_tasks[-1])
    subtask_desc3['response']={'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)

    cot_sc_instruction4='Sub-task 4: Outline the one-loop effective potential approach and write the general expression for m_{H_2}^2 = ∂^2 V_eff/∂H_2^2 evaluated at the vacuum.'
    N=self.max_sc
    cot_agents4=[LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers4=[]
    thinkingmapping4={}
    answermapping4={}
    subtask_desc4={'subtask_id':'subtask_4','instruction':cot_sc_instruction4,'context':['user query','subtask_3 thinking','subtask_3 answer'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking4_i,answer4_i=await cot_agents4[i]([taskInfo,thinking3,answer3],cot_sc_instruction4,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents4[i].id}, outlining one-loop potential, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
        possible_answers4.append(answer4_i.content)
        thinkingmapping4[answer4_i.content]=thinking4_i
        answermapping4[answer4_i.content]=answer4_i
    answer4_content=Counter(possible_answers4).most_common(1)[0][0]
    thinking4=thinkingmapping4[answer4_content]
    answer4=answermapping4[answer4_content]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ',sub_tasks[-1])
    subtask_desc4['response']={'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)

    cot_sc_instruction5='Sub-task 5: Derive the generic structure m_{H_2}^2 = 1/[8π^2(x^2+v^2)] · Σ_i α_i M_i^4, explaining the origin of the prefactor.'
    N=self.max_sc
    cot_agents5=[LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers5=[]
    thinkingmapping5={}
    answermapping5={}
    subtask_desc5={'subtask_id':'subtask_5','instruction':cot_sc_instruction5,'context':['user query','subtask_4 thinking','subtask_4 answer'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking5_i,answer5_i=await cot_agents5[i]([taskInfo,thinking4,answer4],cot_sc_instruction5,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents5[i].id}, deriving generic structure, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
        possible_answers5.append(answer5_i.content)
        thinkingmapping5[answer5_i.content]=thinking5_i
        answermapping5[answer5_i.content]=answer5_i
    answer5_content=Counter(possible_answers5).most_common(1)[0][0]
    thinking5=thinkingmapping5[answer5_content]
    answer5=answermapping5[answer5_content]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Step 5: ',sub_tasks[-1])
    subtask_desc5['response']={'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)

    cot_sc_instruction6='Sub-task 6: List all particle species contributing in loops: h_1, W, Z, H^±, H^0, A^0, N_i, and the top quark if applicable.'
    N=self.max_sc
    cot_agents6=[LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers6=[]
    thinkingmapping6={}
    answermapping6={}
    subtask_desc6={'subtask_id':'subtask_6','instruction':cot_sc_instruction6,'context':['user query','subtask_5 thinking','subtask_5 answer'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking6_i,answer6_i=await cot_agents6[i]([taskInfo,thinking5,answer5],cot_sc_instruction6,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents6[i].id}, listing loop species, thinking: {thinking6_i.content}; answer: {answer6_i.content}')
        possible_answers6.append(answer6_i.content)
        thinkingmapping6[answer6_i.content]=thinking6_i
        answermapping6[answer6_i.content]=answer6_i
    answer6_content=Counter(possible_answers6).most_common(1)[0][0]
    thinking6=thinkingmapping6[answer6_content]
    answer6=answermapping6[answer6_content]
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Step 6: ',sub_tasks[-1])
    subtask_desc6['response']={'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)

    cot_reflect_instruction7='Sub-task 7: Determine the sign of each contribution based on bosonic (positive) or fermionic (negative) statistics and assign correct α_i indices.'
    cot_agent_ref7=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    critic_agent7=LLMAgentBase(['feedback','correct'],'Critic Agent',model=self.node_model,temperature=0.0)
    N_max=self.max_round
    cot_inputs7=[taskInfo,thinking5,answer5,thinking6,answer6]
    subtask_desc7={'subtask_id':'subtask_7','instruction':cot_reflect_instruction7,'context':['user query','subtask_5 thinking','subtask_5 answer','subtask_6 thinking','subtask_6 answer'],'agent_collaboration':'Reflexion'}
    thinking7,answer7=await cot_agent_ref7(cot_inputs7,cot_reflect_instruction7,0,is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_ref7.id}, determining signs, thinking: {thinking7.content}; answer: {answer7.content}')
    for i in range(N_max):
        feedback7,correct7=await critic_agent7([taskInfo,thinking7,answer7],'Please review the sign assignment and provide its limitations.',i,is_sub_task=True)
        agents.append(f'Critic agent {critic_agent7.id}, review round {i}, feedback: {feedback7.content}; correct: {correct7.content}')
        if correct7.content=='True':
            break
        cot_inputs7.extend([thinking7,answer7,feedback7])
        thinking7,answer7=await cot_agent_ref7(cot_inputs7,cot_reflect_instruction7,i+1,is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_ref7.id}, refining sign determination, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print('Step 7: ',sub_tasks[-1])
    subtask_desc7['response']={'thinking':thinking7,'answer':answer7}
    logs.append(subtask_desc7)

    cot_sc_instruction8='Sub-task 8: Assemble the full expression for m_{H_2}^2 including all identified contributions and their signs over 8π^2(x^2+v^2).'
    N=self.max_sc
    cot_agents8=[LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers8=[]
    thinkingmapping8={}
    answermapping8={}
    subtask_desc8={'subtask_id':'subtask_8','instruction':cot_sc_instruction8,'context':['user query','subtask_7 thinking','subtask_7 answer'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking8_i,answer8_i=await cot_agents8[i]([taskInfo,thinking7,answer7],cot_sc_instruction8,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents8[i].id}, assembling full expression, thinking: {thinking8_i.content}; answer: {answer8_i.content}')
        possible_answers8.append(answer8_i.content)
        thinkingmapping8[answer8_i.content]=thinking8_i
        answermapping8[answer8_i.content]=answer8_i
    answer8_content=Counter(possible_answers8).most_common(1)[0][0]
    thinking8=thinkingmapping8[answer8_content]
    answer8=answermapping8[answer8_content]
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print('Step 8: ',sub_tasks[-1])
    subtask_desc8['response']={'thinking':thinking8,'answer':answer8}
    logs.append(subtask_desc8)

    cot_reflect_instruction9='Sub-task 9: Compare the assembled expression with choices A, B, C, and D to find the exact structural match.'
    cot_agent_ref9=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    critic_agent9=LLMAgentBase(['feedback','correct'],'Critic Agent',model=self.node_model,temperature=0.0)
    N_max=self.max_round
    cot_inputs9=[taskInfo,thinking8,answer8]
    subtask_desc9={'subtask_id':'subtask_9','instruction':cot_reflect_instruction9,'context':['user query','subtask_8 thinking','subtask_8 answer'],'agent_collaboration':'Reflexion'}
    thinking9,answer9=await cot_agent_ref9(cot_inputs9,cot_reflect_instruction9,0,is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_ref9.id}, comparing expressions, thinking: {thinking9.content}; answer: {answer9.content}')
    for i in range(N_max):
        feedback9,correct9=await critic_agent9([taskInfo,thinking9,answer9],'Please review the comparison and identify any mismatch.',i,is_sub_task=True)
        agents.append(f'Critic agent {critic_agent9.id}, review round {i}, feedback: {feedback9.content}; correct: {correct9.content}')
        if correct9.content=='True':
            break
        cot_inputs9.extend([thinking9,answer9,feedback9])
        thinking9,answer9=await cot_agent_ref9(cot_inputs9,cot_reflect_instruction9,i+1,is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_ref9.id}, refining comparison, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    print('Step 9: ',sub_tasks[-1])
    subtask_desc9['response']={'thinking':thinking9,'answer':answer9}
    logs.append(subtask_desc9)

    cot_instruction10='Sub-task 10: Select and output the correct multiple-choice letter corresponding to the matching formula.'
    cot_agent10=LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc10={'subtask_id':'subtask_10','instruction':cot_instruction10,'context':['user query','subtask_9 thinking','subtask_9 answer'],'agent_collaboration':'CoT'}
    thinking10,answer10=await cot_agent10([taskInfo,thinking9,answer9],cot_instruction10,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent10.id}, selecting final answer, thinking: {thinking10.content}; answer: {answer10.content}')
    sub_tasks.append(f'Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}')
    print('Step 10: ',sub_tasks[-1])
    subtask_desc10['response']={'thinking':thinking10,'answer':answer10}
    logs.append(subtask_desc10)

    final_answer=await self.make_final_answer(thinking10,answer10,sub_tasks,agents)
    return final_answer,logs