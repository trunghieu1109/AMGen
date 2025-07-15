async def forward_165(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_0 = "Sub-task 1: Extract and summarize the defining features of the extended Standard Model Lagrangian, including field content, vacuum expectation values, and relevant interactions that influence the pseudo-Goldstone boson mass. Ensure clarity on the combined VEV notation and the role of each field in mass generation to avoid ambiguity in later stages." + debate_instr
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking0 = [[] for _ in range(N_max_0)]
    all_answer0 = [[] for _ in range(N_max_0)]
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instruction_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking0[r-1] + all_answer0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instruction_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing Lagrangian and VEVs, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking0[r].append(thinking0)
            all_answer0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking0[-1] + all_answer0[-1], "Sub-task 1: Synthesize and finalize the summary of the extended Standard Model Lagrangian and VEVs." + debate_instr, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing Lagrangian summary, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = "Sub-task 1: Derive the one-loop effective potential (Coleman–Weinberg potential) for the scalar sector involving phi, H, and S fields from first principles. Explicitly identify all fields with nonzero couplings to phi and H, including gauge bosons, scalar bosons, singlet fermions N_i, and the Standard Model top quark. This subtask addresses the previous failure to include the dominant top-quark loop and ensures all relevant contributions are accounted for with correct signs."
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking1_1, answer1_1 = await cot_agents_1_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, deriving one-loop effective potential, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        possible_answers_1_1.append(answer1_1)
        possible_thinkings_1_1.append(thinking1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + possible_thinkings_1_1 + possible_answers_1_1, "Sub-task 1: Synthesize the one-loop effective potential derivation.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing one-loop effective potential, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = "Sub-task 2: Compute the second derivative of the one-loop effective potential with respect to the pseudo-Goldstone boson field H_2 to obtain the approximate mass squared formula. This step must explicitly show the negative contribution from the top-quark loop and positive contributions from bosonic loops, ensuring the correct structure and sign conventions of the radiative corrections."
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking1_1.content, answer1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking1_2, answer1_2 = await cot_agents_1_2[i]([taskInfo, thinking1_1, answer1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, computing second derivative for mass formula, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2)
        possible_thinkings_1_2.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo, thinking1_1, answer1_1] + possible_thinkings_1_2 + possible_answers_1_2, "Sub-task 2: Synthesize the mass squared formula from the second derivative.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing mass squared formula, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc1_2)
    print("Step 1.2: ", sub_tasks[-1])

    reflect_inst_1_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_3 = "Sub-task 3: Verify the normalization factor and dimensional consistency of the derived mass formula. Confirm that the combined VEV factor (x^2 + v^2) appears in the denominator with the correct loop factor 8pi^2, and rule out incorrect placements such as numerator positions. This subtask addresses previous oversights in normalization and ensures the formula aligns with standard effective potential calculations." + reflect_inst_1_3
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2]
    subtask_desc1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking1_1.content, answer1_1.content, thinking1_2.content, answer1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking1_3, answer1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, verifying normalization and dimensional consistency, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
    for i in range(N_max_1_3):
        critic_inst_1_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback1_3, correct1_3 = await critic_agent_1_3([taskInfo, thinking1_3, answer1_3], "Please review and provide the limitations of provided solutions" + critic_inst_1_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback1_3.content}; answer: {correct1_3.content}")
        if correct1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking1_3, answer1_3, feedback1_3])
        thinking1_3, answer1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining normalization verification, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    subtask_desc1_3['response'] = {"thinking": thinking1_3, "answer": answer1_3}
    logs.append(subtask_desc1_3)
    print("Step 1.3: ", sub_tasks[-1])

    reflect_inst_2_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = "Sub-task 1: Integrate and assemble the contributions from all relevant particles (gauge bosons, scalar bosons including charged and neutral Higgs states, singlet fermions, and the top quark) into a coherent composite formula for the pseudo-Goldstone boson mass squared. Ensure correct coefficients alpha_i, signs, and summations are applied, explicitly justifying inclusion or exclusion of each term based on the model's physics and previous derivations." + reflect_inst_2_1
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking1_3, answer1_3, thinking0, answer0]
    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking1_3.content, answer1_3.content, thinking0.content, answer0.content],
        "agent_collaboration": "Reflexion"
    }
    thinking2_1, answer2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, assembling composite mass formula, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    for i in range(N_max_2_1):
        critic_inst_2_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback2_1, correct2_1 = await critic_agent_2_1([taskInfo, thinking2_1, answer2_1], "Please review and provide the limitations of provided solutions" + critic_inst_2_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback2_1.content}; answer: {correct2_1.content}")
        if correct2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking2_1, answer2_1, feedback2_1])
        thinking2_1, answer2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining composite formula, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflect_inst_3_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3_1 = "Sub-task 1: Critically evaluate the four candidate formulae for the pseudo-Goldstone boson mass squared against the derived theoretical expression. Use a multi-agent Reflexion pattern to challenge assumptions, reconcile discrepancies, and prioritize the formula that correctly includes the dominant top-quark negative contribution, proper normalization, and all relevant scalar and fermionic terms. Avoid premature consensus by enforcing rigorous cross-validation of physical principles and dimensional analysis." + reflect_inst_3_1
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1 = self.max_round
    cot_inputs_3_1 = [taskInfo, thinking2_1, answer2_1]
    subtask_desc3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_reflect_instruction_3_1,
        "context": ["user query", thinking2_1.content, answer2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3_1, answer3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, evaluating candidate formulae, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    for i in range(N_max_3_1):
        critic_inst_3_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback3_1, correct3_1 = await critic_agent_3_1([taskInfo, thinking3_1, answer3_1], "Please review and provide the limitations of provided solutions" + critic_inst_3_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, providing feedback, thinking: {feedback3_1.content}; answer: {correct3_1.content}")
        if correct3_1.content == "True":
            break
        cot_inputs_3_1.extend([thinking3_1, answer3_1, feedback3_1])
        thinking3_1, answer3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining candidate evaluation, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_1, answer3_1, sub_tasks, agents)
    return final_answer, logs
