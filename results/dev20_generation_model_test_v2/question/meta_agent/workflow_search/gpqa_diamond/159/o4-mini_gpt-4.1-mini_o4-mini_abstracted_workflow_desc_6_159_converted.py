async def forward_159(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0 - Sub-task 1
    sc_instruction1 = 'Sub-task 1: Summarize the aperture geometry and show that as N→∞ the apothem a becomes the radius of a circular aperture.'
    cot_agents1 = [LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':sc_instruction1,'context':['user query'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents1:
        thinking_i, answer_i = await agent([taskInfo], sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision1 = LLMAgentBase(['thinking','answer'],'Final Decision Agent',model=self.node_model,temperature=0.0)
    final_instr1 = 'Sub-task 1: Synthesize and choose the most consistent answer for subtask 1. Given all the above thinking and answers, find the most consistent and correct solution for subtask 1.'
    thinking1, answer1 = await final_decision1([taskInfo] + possible_thinkings1 + possible_answers1, final_instr1, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    # Stage 0 - Sub-task 2
    sc_instruction2 = 'Sub-task 2: Establish the Fraunhofer diffraction amplitude for a circular aperture of radius a and relate the intensity minima to the zeros of the Bessel function J1.'
    cot_agents2 = [LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':sc_instruction2,'context':['user query','thinking of subtask 1','answer of subtask 1'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents2:
        thinking_i, answer_i = await agent([taskInfo,thinking1,answer1], sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings2.append(thinking_i)
        possible_answers2.append(answer_i)
    final_decision2 = LLMAgentBase(['thinking','answer'],'Final Decision Agent',model=self.node_model,temperature=0.0)
    final_instr2 = 'Sub-task 2: Synthesize and choose the most consistent answer for subtask 2. Given all the above thinking and answers, find the most consistent and correct solution for subtask 2.'
    thinking2, answer2 = await final_decision2([taskInfo,thinking1,answer1] + possible_thinkings2 + possible_answers2, final_instr2, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    # Stage 1 - Sub-task 3
    sc_instruction3 = 'Sub-task 3: Identify the first two positive roots x1 and x2 of J1(x)=0 and express their angular positions θm = (xm·λ)/(2πa) under the small-angle approximation.'
    cot_agents3 = [LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':sc_instruction3,'context':['user query','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents3:
        thinking_i, answer_i = await agent([taskInfo,thinking2,answer2], sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings3.append(thinking_i)
        possible_answers3.append(answer_i)
    final_decision3 = LLMAgentBase(['thinking','answer'],'Final Decision Agent',model=self.node_model,temperature=0.0)
    final_instr3 = 'Sub-task 3: Synthesize and choose the most consistent answer for subtask 3. Given all the above thinking and answers, find the most consistent and correct solution for subtask 3.'
    thinking3, answer3 = await final_decision3([taskInfo,thinking2,answer2] + possible_thinkings3 + possible_answers3, final_instr3, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    # Stage 1 - Sub-task 4
    sc_instruction4 = 'Sub-task 4: Compute the angular separation Δθ = (x2 – x1)·λ/(2πa) numerically using the tabulated roots.'
    cot_agents4 = [LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':sc_instruction4,'context':['user query','thinking of subtask 3','answer of subtask 3'],'agent_collaboration':'SC_CoT'}
    for agent in cot_agents4:
        thinking_i, answer_i = await agent([taskInfo,thinking3,answer3], sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings4.append(thinking_i)
        possible_answers4.append(answer_i)
    final_decision4 = LLMAgentBase(['thinking','answer'],'Final Decision Agent',model=self.node_model,temperature=0.0)
    final_instr4 = 'Sub-task 4: Synthesize and choose the most consistent answer for subtask 4. Given all the above thinking and answers, find the most consistent and correct solution for subtask 4.'
    thinking4, answer4 = await final_decision4([taskInfo,thinking3,answer3] + possible_thinkings4 + possible_answers4, final_instr4, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    # Stage 2 - Sub-task 5
    reflect_inst = 'Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.'
    cot_reflect_instruction = 'Sub-task 5: Compare the computed Δθ value with the provided choices and select the correct option. ' + reflect_inst
    critic_inst = 'Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly True in correct'
    cot_agent5 = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    critic_agent5 = LLMAgentBase(['feedback','correct'],'Critic Agent',model=self.node_model,temperature=0.0)
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':cot_reflect_instruction,'context':['user query','thinking of subtask 4','answer of subtask 4'],'agent_collaboration':'Reflexion'}
    cot_inputs5 = [taskInfo,thinking1,answer1,thinking2,answer2,thinking3,answer3,thinking4,answer4]
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for _ in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo,thinking5,answer5], critic_inst, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == 'True':
            break
        cot_inputs5 += [thinking5,answer5,feedback5]
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs