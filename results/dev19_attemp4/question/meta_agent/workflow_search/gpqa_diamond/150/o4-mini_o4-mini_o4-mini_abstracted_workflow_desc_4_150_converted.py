async def forward_150(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: Normalize the state vector (SC-CoT)
    cot_sc_instruction1 = "Sub-task 1: Normalize the state vector ψ = (-1, 2, 1)^T by computing its norm and dividing each component by that norm."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, normalizing vector, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent normalization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2: Eigenvalue decomposition of P (Debate)
    debate_instr2 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction2 = "Sub-task 2: Perform the eigenvalue decomposition of P to identify the eigenvectors corresponding to eigenvalue zero and characterize the zero-eigenspace." + debate_instr2
    debate_agents2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N2 = self.max_round
    all_thinking2 = [[] for _ in range(N2)]
    all_answer2 = [[] for _ in range(N2)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":debate_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"Debate"}
    for r in range(N2):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                inputs2 = [taskInfo, thinking1, answer1]
            else:
                inputs2 = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
            thinking2_i, answer2_i = await agent(inputs2, debate_instruction2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
            all_thinking2[r].append(thinking2_i)
            all_answer2[r].append(answer2_i)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: " + final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3: Construct projection operator (SC-CoT)
    cot_sc_instruction3 = "Sub-task 3: Construct the projection operator Π0 onto the zero-eigenvalue subspace using the eigenvectors found in subtask_2."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, building projector, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the most consistent projection operator.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 4: Apply projector and compute probability (Reflexion)
    reflect_inst4 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction4 = "Sub-task 4: Apply Π0 to the normalized state from subtask_1, compute the squared norm of the projection, and thus obtain the probability of measuring 0." + reflect_inst4
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N4 = self.max_round
    cot_inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction4,"context":["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, initial computation, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N4):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], "Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refinement, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs