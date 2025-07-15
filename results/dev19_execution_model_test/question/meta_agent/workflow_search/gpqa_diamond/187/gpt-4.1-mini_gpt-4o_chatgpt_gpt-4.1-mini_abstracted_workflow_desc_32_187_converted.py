async def forward_187(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = "Sub-task 1: Extract and summarize all given information about the rhombohedral lattice parameters, angles, and the (111) plane indices. Critically evaluate and clarify assumptions such as the meaning of 'interatomic distance' and whether it corresponds directly to the lattice parameter 'a'. Highlight any ambiguities or potential pitfalls in interpreting the problem statement to avoid incorrect assumptions in later calculations. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_stage0 = self.max_round
    all_thinking_stage0 = [[] for _ in range(N_rounds_stage0)]
    all_answer_stage0 = [[] for _ in range(N_rounds_stage0)]
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_stage0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_stage0):
        for i, agent in enumerate(debate_agents_stage0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instr_stage0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_stage0[r-1] + all_answer_stage0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instr_stage0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking_stage0[r].append(thinking0)
            all_answer_stage0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_stage0[-1] + all_answer_stage0[-1], "Sub-task 1: Extract and clarify problem data." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 0, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Stage 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_stage1_1 = "Sub-task 1: Identify and derive the correct formula for the interplanar spacing d_hkl in a rhombohedral lattice with arbitrary lattice angle alpha. Include referencing crystallographic literature or standard formulae involving the reciprocal lattice metric tensor, ensuring the formula accounts for the non-orthogonal lattice angles and is valid for acute angles such as 30 degrees. Avoid oversimplified or approximate formulas. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_stage1_1 = self.max_round
    all_thinking_stage1_1 = [[] for _ in range(N_rounds_stage1_1)]
    all_answer_stage1_1 = [[] for _ in range(N_rounds_stage1_1)]
    subtask_desc1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_stage1_1):
        for i, agent in enumerate(debate_agents_stage1_1):
            if r == 0:
                thinking1_1, answer1_1 = await agent([taskInfo, thinking0, answer0], debate_instr_stage1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking0, answer0] + all_thinking_stage1_1[r-1] + all_answer_stage1_1[r-1]
                thinking1_1, answer1_1 = await agent(input_infos_1_1, debate_instr_stage1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id} stage 1.1, round {r}, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
            all_thinking_stage1_1[r].append(thinking1_1)
            all_answer_stage1_1[r].append(answer1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + all_thinking_stage1_1[-1] + all_answer_stage1_1[-1], "Sub-task 1.1: Derive formula for interplanar spacing." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 1.1, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Stage 1.1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_stage1_2 = "Sub-task 2: Validate the derived formula by cross-checking it against known special cases (e.g., cubic or hexagonal limits) or authoritative crystallography sources. Prevent propagation of incorrect formulas by requiring explicit verification and justification of the formula's applicability to the given lattice parameters."
    N_sc = self.max_sc
    cot_agents_stage1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage1_2,
        "context": ["user query", thinking0, answer0, thinking1_1, answer1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1_2, answer1_2 = await cot_agents_stage1_2[i]([taskInfo, thinking0, answer0, thinking1_1, answer1_1], cot_sc_instruction_stage1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_2[i].id} stage 1.2, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2)
        possible_thinkings_1_2.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo, thinking0, answer0, thinking1_1, answer1_1] + possible_thinkings_1_2 + possible_answers_1_2, "Sub-task 1.2: Validate formula." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 1.2, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Stage 1.2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_stage2_1 = "Sub-task 1: Substitute the given numerical values (a = 10 Angstrom, alpha = 30 degrees, h = k = l = 1) into the validated formula and perform careful symbolic and numeric simplification. Include explicit numeric checks of intermediate steps, such as computing radicands and square roots accurately, to avoid arithmetic errors."
    N_sc_stage2_1 = self.max_sc
    cot_agents_stage2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_stage2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_stage2_1,
        "context": ["user query", thinking0, answer0, thinking1_2, answer1_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_stage2_1):
        thinking2_1, answer2_1 = await cot_agents_stage2_1[i]([taskInfo, thinking0, answer0, thinking1_2, answer1_2], cot_sc_instruction_stage2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2_1[i].id} stage 2.1, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
        possible_answers_2_1.append(answer2_1)
        possible_thinkings_2_1.append(thinking2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1([taskInfo, thinking0, answer0, thinking1_2, answer1_2] + possible_thinkings_2_1 + possible_answers_2_1, "Sub-task 2.1: Numeric substitution and simplification." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 2.1, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Stage 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflect_inst_stage2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage2_2 = "Sub-task 2: Perform unit consistency checks and confirm that the computed interplanar distance is physically reasonable given the lattice parameters and angles. Include a sanity check comparing the computed value against typical interplanar distances for rhombohedral lattices with similar parameters to detect any outliers or calculation errors." + reflect_inst_stage2_2
    cot_agent_stage2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage2_2 = self.max_round
    cot_inputs_stage2_2 = [taskInfo, thinking0, answer0, thinking1_2, answer1_2, thinking2_1, answer2_1]
    subtask_desc2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_stage2_2,
        "context": ["user query", thinking0, answer0, thinking1_2, answer1_2, thinking2_1, answer2_1],
        "agent_collaboration": "Reflexion"
    }
    thinking2_2, answer2_2 = await cot_agent_stage2_2(cot_inputs_stage2_2, cot_reflect_instruction_stage2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent stage 2.2, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    for i in range(N_max_stage2_2):
        feedback, correct = await critic_agent_stage2_2([taskInfo, thinking2_2, answer2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent stage 2.2, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage2_2.extend([thinking2_2, answer2_2, feedback])
        thinking2_2, answer2_2 = await cot_agent_stage2_2(cot_inputs_stage2_2, cot_reflect_instruction_stage2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent stage 2.2 refining, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Stage 2.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instr_stage3_1 = "Sub-task 1: Compare the accurately computed interplanar distance with all provided answer choices: 9.54, 8.95, 9.08, and 10.05 Angstrom. Calculate the absolute differences to select the closest matching value. Document the reasoning behind the choice to ensure transparency and prevent bias from previous incorrect approximations. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_stage3_1 = self.max_round
    all_thinking_stage3_1 = [[] for _ in range(N_rounds_stage3_1)]
    all_answer_stage3_1 = [[] for _ in range(N_rounds_stage3_1)]
    subtask_desc3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3_1,
        "context": ["user query", thinking2_2, answer2_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_stage3_1):
        for i, agent in enumerate(debate_agents_stage3_1):
            if r == 0:
                thinking3_1, answer3_1 = await agent([taskInfo, thinking2_2, answer2_2], debate_instr_stage3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking2_2, answer2_2] + all_thinking_stage3_1[r-1] + all_answer_stage3_1[r-1]
                thinking3_1, answer3_1 = await agent(input_infos_3_1, debate_instr_stage3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id} stage 3.1, round {r}, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
            all_thinking_stage3_1[r].append(thinking3_1)
            all_answer_stage3_1[r].append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking2_2, answer2_2] + all_thinking_stage3_1[-1] + all_answer_stage3_1[-1], "Sub-task 3.1: Select closest answer choice." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 3.1, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Stage 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_1, answer3_1, sub_tasks, agents)
    return final_answer, logs
