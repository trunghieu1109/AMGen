async def forward_157(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0: SC_CoT for extraction
    N0 = self.max_sc
    sc0_instruction = 'Sub-task 1: Extract and classify the key molecular components and mutations from the query, identifying the transactivation domain, dimerization domain, phosphorylation requirement, mutation X properties, and mutation Y properties.'
    sc0_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N0)]
    poss0_think = []
    poss0_ans = []
    for agent in sc0_agents:
        thinking0, answer0 = await agent([taskInfo], sc0_instruction, is_sub_task=True)
        agents.append(f"SC_CoT agent {agent.id}, extracting components, thinking: {thinking0.content}; answer: {answer0.content}")
        poss0_think.append(thinking0)
        poss0_ans.append(answer0)
    final0 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final0([taskInfo] + poss0_think + poss0_ans, 'Sub-task 1: Synthesize and choose the most consistent extraction and classification of key components.', is_sub_task=True)
    agents.append(f"Final Decision agent {final0.id}, synthesizing components, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({'subtask_id':'subtask_1','instruction':sc0_instruction,'context':['user query'],'agent_collaboration':'SC_CoT','response':{'thinking':thinking1,'answer':answer1})
    print('Step 1: ', sub_tasks[-1])
    # Stage 1: SC_CoT for mechanistic assessment
    N1 = self.max_sc
    sc1_instruction = 'Sub-task 2: Assess the mechanistic effect of a heterozygous dominant-negative mutation in the dimerization domain on wild-type protein function.'
    sc1_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N1)]
    poss1_think = []
    poss1_ans = []
    for agent in sc1_agents:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], sc1_instruction, is_sub_task=True)
        agents.append(f"SC_CoT agent {agent.id}, assessing mechanism, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        poss1_think.append(thinking2_i)
        poss1_ans.append(answer2_i)
    final1 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final1([taskInfo, thinking1, answer1] + poss1_think + poss1_ans, 'Sub-task 2: Synthesize and choose the most consistent mechanistic effect.', is_sub_task=True)
    agents.append(f"Final Decision agent {final1.id}, synthesizing mechanism, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({'subtask_id':'subtask_2','instruction':sc1_instruction,'context':['user query','thinking of subtask 1','answer of subtask 1'],'agent_collaboration':'SC_CoT','response':{'thinking':thinking2,'answer':answer2})
    print('Step 2: ', sub_tasks[-1])
    # Stage 2: Debate to compare phenotype options
    debate_instr = 'Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.'
    debate_instruction = 'Sub-task 3: Compare each provided molecular phenotype option against the predicted dominant-negative mechanism to evaluate consistency.' + debate_instr
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R = self.max_round
    all_think3 = [[] for _ in range(R)]
    all_ans3 = [[] for _ in range(R)]
    for r in range(R):
        for agent in debate_agents:
            if r == 0:
                t3, a3 = await agent([taskInfo, thinking2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_think3[r-1] + all_ans3[r-1]
                t3, a3 = await agent(inputs3, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t3.content}; answer: {a3.content}")
            all_think3[r].append(t3)
            all_ans3[r].append(a3)
    final3 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    final3_instr = 'Sub-task 3: Compare and choose the most consistent molecular phenotype.' + 'Given all the above thinking and answers, reason over them carefully and provide a final answer.'
    thinking3, answer3 = await final3([taskInfo, thinking2, answer2] + all_think3[-1] + all_ans3[-1], final3_instr, is_sub_task=True)
    agents.append(f"Final Decision agent {final3.id}, finalizing comparison, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({'subtask_id':'subtask_3','instruction':debate_instruction,'context':['user query','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'Debate','response':{'thinking':thinking3,'answer':answer3})
    print('Step 3: ', sub_tasks[-1])
    # Stage 3: Reflexion for final rationale
    reflect_inst = 'Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.'
    reflex_instruction = 'Sub-task 4: Select the most likely molecular phenotype and provide a concise rationale.' + reflect_inst
    cot4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    for i in range(self.max_round):
        thinking4, answer4 = await cot4(inputs4, reflex_instruction, i, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly True in correct"
        feedback, correct = await critic([taskInfo, thinking4, answer4], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic.id}, feedback: {feedback.content}; answer: {correct.content}")
        if correct.content == 'True':
            break
        inputs4.extend([thinking4, answer4, feedback])
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append({'subtask_id':'subtask_4','instruction':reflex_instruction,'context':['user query','thinking of subtask 1','answer of subtask 1','thinking of subtask 2','answer of subtask 2','thinking of subtask 3','answer of subtask 3'],'agent_collaboration':'Reflexion','response':{'thinking':thinking4,'answer':answer4})
    print('Step 4: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs