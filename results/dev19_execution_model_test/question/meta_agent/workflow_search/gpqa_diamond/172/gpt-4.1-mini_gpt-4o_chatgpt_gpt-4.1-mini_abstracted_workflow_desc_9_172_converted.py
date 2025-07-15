async def forward_172(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Extract and summarize given parameters (Debate)
    debate_instr_stage1 = "Sub-task 1: Extract and summarize all given physical parameters, constants, and problem conditions relevant to the uncertainty calculation, ensuring clarity on assumptions such as electron rest mass, speed of light, and unit conversions to avoid ambiguity in later calculations. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage1 = self.max_round
    all_thinking_stage1 = [[] for _ in range(N_max_stage1)]
    all_answer_stage1 = [[] for _ in range(N_max_stage1)]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1):
        for i, agent in enumerate(debate_agents_stage1):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_stage1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_stage1[r-1] + all_answer_stage1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage1[r].append(thinking)
            all_answer_stage1[r].append(answer)
    final_decision_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_stage1([taskInfo] + all_thinking_stage1[-1] + all_answer_stage1[-1], "Sub-task 1: Extract and summarize parameters. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Calculate minimum uncertainty in momentum Δp from Δx (SC_CoT)
    cot_sc_instruction_stage1_2 = "Sub-task 2: Based on the output from Sub-task 1, apply the Heisenberg uncertainty principle to calculate the minimum uncertainty in momentum Δp from the given uncertainty in position Δx, explicitly stating the formula Δx·Δp ≥ ħ/2 and performing unit consistency checks to avoid errors in Δp estimation."
    N_sc = self.max_sc
    cot_agents_stage1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage1_2 = []
    possible_thinkings_stage1_2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage1_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_stage1_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_stage1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_2[i].id}, calculating Δp, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_stage1_2.append(answer2)
        possible_thinkings_stage1_2.append(thinking2)
    final_decision_agent_stage1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_stage1_2([taskInfo, thinking1, answer1] + possible_thinkings_stage1_2 + possible_answers_stage1_2, "Sub-task 2: Synthesize and choose the most consistent and correct Δp calculation.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Compute relativistic parameters (SC_CoT)
    cot_sc_instruction_stage2_1 = "Sub-task 1: Compute the relativistic parameters: calculate the Lorentz factor γ = 1 / sqrt(1 - (v/c)^2), relativistic momentum p = γ m₀ v, and total energy E = γ m₀ c^2, explicitly using correct constants and units. This subtask addresses previous errors where relativistic momentum and energy were inconsistently or incorrectly calculated."
    cot_agents_stage2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage2_1 = []
    possible_thinkings_stage2_1 = []
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_stage2_1,
        "context": ["user query", thinking2, answer2, thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_stage2_1[i]([taskInfo, thinking2, answer2, thinking1, answer1], cot_sc_instruction_stage2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2_1[i].id}, computing relativistic parameters, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_stage2_1.append(answer3)
        possible_thinkings_stage2_1.append(thinking3)
    final_decision_agent_stage2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_stage2_1([taskInfo, thinking2, answer2, thinking1, answer1] + possible_thinkings_stage2_1 + possible_answers_stage2_1, "Sub-task 1: Synthesize and choose the most consistent relativistic parameters.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Estimate minimum uncertainty in kinetic energy ΔE (Debate)
    debate_instr_stage2_2 = "Sub-task 2: Estimate the minimum uncertainty in kinetic energy ΔE from the uncertainty in momentum Δp using the correct relativistic relationship ΔE ≈ (∂E/∂p)·Δp, where ∂E/∂p = p c^2 / E = v. Verify unit consistency, confirm that Δp is sufficiently small for linear approximation, and if not, perform a numerical evaluation of ΔE = E(p + Δp) - E(p) to avoid oversimplification errors identified previously. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage2_2 = self.max_round
    all_thinking_stage2_2 = [[] for _ in range(N_max_stage2_2)]
    all_answer_stage2_2 = [[] for _ in range(N_max_stage2_2)]
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": debate_instr_stage2_2,
        "context": ["user query", thinking3, answer3, thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage2_2):
        for i, agent in enumerate(debate_agents_stage2_2):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3, thinking2, answer2], debate_instr_stage2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3, thinking2, answer2] + all_thinking_stage2_2[r-1] + all_answer_stage2_2[r-1]
                thinking4, answer4 = await agent(input_infos, debate_instr_stage2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, estimating ΔE, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_stage2_2[r].append(thinking4)
            all_answer_stage2_2[r].append(answer4)
    final_decision_agent_stage2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_stage2_2([taskInfo, thinking3, answer3, thinking2, answer2] + all_thinking_stage2_2[-1] + all_answer_stage2_2[-1], "Sub-task 2: Synthesize and choose the most consistent ΔE estimation.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Evaluate ΔE against choices and select closest (Debate)
    debate_instr_stage3_1 = "Sub-task 1: Evaluate the calculated minimum uncertainty in energy ΔE against the provided multiple-choice options, select the closest match, and justify the choice with clear reasoning. Critically assess the final result's physical plausibility and consistency with relativistic quantum mechanics principles. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage3_1 = self.max_round
    all_thinking_stage3_1 = [[] for _ in range(N_max_stage3_1)]
    all_answer_stage3_1 = [[] for _ in range(N_max_stage3_1)]
    subtask_desc5 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3_1,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage3_1):
        for i, agent in enumerate(debate_agents_stage3_1):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instr_stage3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking4, answer4] + all_thinking_stage3_1[r-1] + all_answer_stage3_1[r-1]
                thinking5, answer5 = await agent(input_infos, debate_instr_stage3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating final ΔE, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_stage3_1[r].append(thinking5)
            all_answer_stage3_1[r].append(answer5)
    final_decision_agent_stage3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_stage3_1([taskInfo, thinking4, answer4] + all_thinking_stage3_1[-1] + all_answer_stage3_1[-1], "Sub-task 1: Synthesize and choose the best matching choice for ΔE.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 3, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
