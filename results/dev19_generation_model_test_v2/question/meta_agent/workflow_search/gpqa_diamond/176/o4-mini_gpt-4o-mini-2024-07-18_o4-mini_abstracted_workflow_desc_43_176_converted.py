async def forward_176(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: Extract and validate parameters using SC-CoT
    cot_sc_instruction = (
        "Sub-task 1: Extract and validate parameters (R1/R2=1.5, M1/M2=1.5, observed λ_max, "
        "radial velocities v1=0 km/s, v2=700 km/s). Ensure to note need for Doppler correction."
    )
    N1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings_1 = []
    possible_answers_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents_1:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_1.append(thinking)
        possible_answers_1.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Sub-task 1: Synthesize and choose the most consistent extracted parameters."
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2.1: Doppler correction using Debate
    debate_instruction = (
        "Sub-task 2.1: Apply the relativistic Doppler shift correction for Star 2 using λ_emit = λ_obs * sqrt((1 - β)/(1 + β)) with β = v/c. "
        "Debate on correction necessity and non-relativistic limit. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": debate_instruction,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking2[r].append(thinking)
            all_answer2[r].append(answer)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Sub-task 2.1: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2_1, answer2_1 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])
    # Stage 2.2: Compute temperatures using SC-CoT
    cot_sc_instruction2 = (
        "Sub-task 2.2: From the corrected wavelengths, apply Wien’s law λ_max T = b to calculate T1 and T2, highlighting the temperature difference. "
        "Emphasize that prior solutions wrongly assumed equal temperatures by ignoring Doppler."
    )
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings_2 = []
    possible_answers_2 = []
    subtask_desc2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": cot_sc_instruction2,
        "context": ["user query", thinking2_1.content, answer2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents_2:
        thinking, answer = await agent([taskInfo, thinking2_1, answer2_1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_2.append(thinking)
        possible_answers_2.append(answer)
    final_decision_agent_2b = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2b = "Sub-task 2.2: Synthesize and choose the most consistent temperatures."
    thinking2_2, answer2_2 = await final_decision_agent_2b([taskInfo, thinking2_1, answer2_1] + possible_thinkings_2 + possible_answers_2, final_instr2b, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 2.2: ", sub_tasks[-1])
    # Stage 3.1: Compute luminosity ratio using SC-CoT
    cot_sc_instruction3 = (
        "Sub-task 3.1: Compute luminosity ratio L1/L2 via Stefan–Boltzmann law L∝R^2 T^4 using R1/R2=1.5 and the distinct temperatures, reflecting the Doppler-induced temperature correction."
    )
    N3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings_3 = []
    possible_answers_3 = []
    subtask_desc3_1 = {
        "subtask_id": "subtask_3_1",
        "instruction": cot_sc_instruction3,
        "context": ["user query", thinking2_2.content, answer2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents_3:
        thinking, answer = await agent([taskInfo, thinking2_2, answer2_2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_3.append(thinking)
        possible_answers_3.append(answer)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3.1: Synthesize and choose the most consistent luminosity ratio."
    thinking3_1, answer3_1 = await final_decision_agent_3([taskInfo, thinking2_2, answer2_2] + possible_thinkings_3 + possible_answers_3, final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])
    # Stage 3.2: Choose closest option using Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    reflex_instruction = (
        "Sub-task 3.2: Compare computed L1/L2 to choices ∼2.25, ∼2.35, ∼2.32, ∼2.23 and select the closest, ensuring the factor from Doppler-temperature difference is included. "
        + reflect_inst
    )
    cot_reflect_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc3_2 = {
        "subtask_id": "subtask_3_2",
        "instruction": reflex_instruction,
        "context": ["user query", thinking3_1.content, answer3_1.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs = [taskInfo, thinking3_1, answer3_1]
    thinking3_2, answer3_2 = await cot_reflect_agent(cot_inputs, reflex_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    for i in range(self.max_round):
        critic_inst = (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct"
        )
        feedback, correct = await critic_agent([taskInfo, thinking3_2, answer3_2], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3_2, answer3_2, feedback])
        thinking3_2, answer3_2 = await cot_reflect_agent(cot_inputs, reflex_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 3.2: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3_2, answer3_2, sub_tasks, agents)
    return final_answer, logs