async def forward_193(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0: SC_CoT to extract key elements
    cot0_instruction = "Sub-task 1: Extract and classify the key elements: spin variables (S1,S2,S3), energy function E = -J (S1S2 + S1S3 + S2S3), and thermodynamic parameter β = 1/(kT)."
    N0 = self.max_sc
    cot0_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {"subtask_id":"subtask_1","instruction":cot0_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot0_agents:
        thinking0, answer0 = await agent([taskInfo], cot0_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision0 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision0([taskInfo] + possible_thinkings0 + possible_answers0, "Sub-task 1: Synthesize and choose the most consistent and correct solution for the key-element extraction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking":thinking0,"answer":answer0}
    logs.append(subtask_desc0)
    print("Step 1:", sub_tasks[-1])
    # Stage 1: SC_CoT to enumerate configurations and compute energies
    cot1_instruction = "Sub-task 2: Based on the output from Sub-task 1, enumerate all 8 spin configurations for (S1,S2,S3) with values ±1 and compute the energy E = -J (S1S2 + S1S3 + S2S3) for each."
    N1 = self.max_sc
    cot1_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_2","instruction":cot1_instruction,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for agent in cot1_agents:
        thinking1, answer1 = await agent([taskInfo, thinking0, answer0], cot1_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1, "Sub-task 2: Synthesize and choose the most consistent enumeration and energy computation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 2:", sub_tasks[-1])
    # Stage 2: SC_CoT to group by energy, multiplicity, compute Boltzmann weights
    cot2_instruction = "Sub-task 3: Based on the output from Sub-task 2, group configurations by energy levels, determine their multiplicities, and compute the Boltzmann weights e^(-βE) for each group."
    N2 = self.max_sc
    cot2_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_3","instruction":cot2_instruction,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for agent in cot2_agents:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 3: Synthesize and choose the most consistent grouping and weight computation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 3:", sub_tasks[-1])
    # Stage 3.1: SC_CoT to sum contributions and derive Z
    cot3_instruction = "Sub-task 4: Based on the output from Sub-task 3, sum the weighted contributions (multiplicity × weight) to derive the closed-form expression for the partition function Z."
    N3 = self.max_sc
    cot3_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_4","instruction":cot3_instruction,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    for agent in cot3_agents:
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], cot3_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings3.append(thinking3)
        possible_answers3.append(answer3)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 4: Synthesize and choose the most consistent final expression for Z.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 4:", sub_tasks[-1])
    # Stage 3.2: Reflexion to compare with choices
    cot_reflect_instruction = "Sub-task 5: Given previous attempts, compare the derived Z expression with the four choices and select the matching one. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_5","instruction":cot_reflect_instruction,"context":["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], critic_inst, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 5:", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs