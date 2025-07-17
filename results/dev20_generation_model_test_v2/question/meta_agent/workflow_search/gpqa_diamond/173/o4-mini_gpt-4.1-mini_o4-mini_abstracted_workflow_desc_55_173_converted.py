async def forward_173(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: SC-CoT to find m1, m2, Q
    cot_sc_instruction = "Sub-task 1: Determine m1 and m2 symbolically in terms of M, given m1 = 2 m2 and m1 + m2 = 0.99 M. Also express Q = (M - (m1 + m2)) c^2."
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking1, answer1 = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings.append(thinking1)
        possible_answers.append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings + possible_answers, 
        "Sub-task 1: Synthesize and choose the most consistent answer for subtask 1.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2.1: Debate to derive symbolic p equation
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 2.1: Set up the relativistic two-body energy-momentum equations E1 + E2 = M c^2 with p1 = p2 = p, and derive a symbolic equation for p in terms of M, m1, and m2." + debate_instr
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking21 = []
    all_answer21 = []
    subtask_desc21 = {"subtask_id": "subtask_2_1", "instruction": debate_instruction, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "Debate"}
    for agent in debate_agents:
        thinking21, answer21 = await agent([taskInfo, thinking1, answer1], debate_instruction, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking21.content}; answer: {answer21.content}")
        all_thinking21.append(thinking21)
        all_answer21.append(answer21)
    final_decision_agent21 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking21, answer21 = await final_decision_agent21([taskInfo, thinking1, answer1] + all_thinking21 + all_answer21, 
        "Sub-task 2.1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking21.content}; answer - {answer21.content}")
    subtask_desc21['response'] = {"thinking": thinking21, "answer": answer21}
    logs.append(subtask_desc21)
    print("Step 2.1: ", sub_tasks[-1])
    # Stage 2.2: Reflexion to solve for p·c numerically
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2.2: Solve the equation from subtask 2.1 numerically for p c (in GeV) carrying at least four significant figures." + reflect_inst
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs2 = [taskInfo, thinking1, answer1, thinking21, answer21]
    subtask_desc22 = {"subtask_id": "subtask_2_2", "instruction": cot_reflect_instruction, "context": ["user query","thinking1","answer1","thinking2.1","answer2.1"], "agent_collaboration": "Reflexion"}
    thinking22, answer22 = await cot_agent2(cot_inputs2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, thinking: {thinking22.content}; answer: {answer22.content}")
    for i in range(self.max_round):
        feedback22, correct22 = await critic_agent2([taskInfo, thinking22, answer22], 
            "Please review and provide the limitations of the provided solution. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent2.id}, feedback: {feedback22.content}; correct: {correct22.content}")
        if correct22.content == "True":
            break
        cot_inputs2 += [thinking22, answer22, feedback22]
        thinking22, answer22 = await cot_agent2(cot_inputs2, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, thinking: {thinking22.content}; answer: {answer22.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking22.content}; answer - {answer22.content}")
    subtask_desc22['response'] = {"thinking": thinking22, "answer": answer22}
    logs.append(subtask_desc22)
    print("Step 2.2: ", sub_tasks[-1])
    # Stage 3.1: SC-CoT to compute T1_rel and T1_cl
    cot_sc_instruction3 = "Sub-task 3.1: Compute T1_rel = sqrt((p c)^2 + (m1 c^2)^2) - m1 c^2 and T1_cl = (p c)^2 / (2 m1 c^2), carrying at least four significant figures each."  
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc31 = {"subtask_id": "subtask_3_1", "instruction": cot_sc_instruction3, "context": ["user query","thinking1","answer1","thinking22","answer22"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents3:
        thinking31, answer31 = await agent([taskInfo, thinking1, answer1, thinking22, answer22], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking31.content}; answer: {answer31.content}")
        possible_thinkings3.append(thinking31)
        possible_answers3.append(answer31)
    final_decision_agent31 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking31, answer31 = await final_decision_agent31(
        [taskInfo, thinking1, answer1, thinking22, answer22] + possible_thinkings3 + possible_answers3,
        "Sub-task 3.1: Synthesize and choose the most consistent answer for subtask 3.1.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking31.content}; answer - {answer31.content}")
    subtask_desc31['response'] = {"thinking": thinking31, "answer": answer31}
    logs.append(subtask_desc31)
    print("Step 3.1: ", sub_tasks[-1])
    # Stage 3.2: Reflexion to compute ΔT
    reflect_inst3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction3 = "Sub-task 3.2: Compute ΔT = T1_rel - T1_cl directly in one expression, and maintain at least four significant figures." + reflect_inst3
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking22, answer22, thinking31, answer31]
    subtask_desc32 = {"subtask_id": "subtask_3_2", "instruction": cot_reflect_instruction3, "context": ["user query","thinking3.1","answer3.1"], "agent_collaboration": "Reflexion"}
    thinking32, answer32 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking32.content}; answer: {answer32.content}")
    for i in range(self.max_round):
        feedback32, correct32 = await critic_agent3([taskInfo, thinking32, answer32], 
            "Please review and provide the limitations of the provided solution. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback32.content}; correct: {correct32.content}")
        if correct32.content == "True":
            break
        cot_inputs3 += [thinking32, answer32, feedback32]
        thinking32, answer32 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking32.content}; answer: {answer32.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking32.content}; answer - {answer32.content}")
    subtask_desc32['response'] = {"thinking": thinking32, "answer": answer32}
    logs.append(subtask_desc32)
    print("Step 3.2: ", sub_tasks[-1])
    # Stage 4: CoT to convert ΔT to MeV and select choice
    cot_instruction4 = "Sub-task 4: Convert ΔT into MeV and match it to the closest provided choice (2 MeV, 5 MeV, 10 MeV, 20 MeV)."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query","thinking3.2","answer3.2"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking32, answer32], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs