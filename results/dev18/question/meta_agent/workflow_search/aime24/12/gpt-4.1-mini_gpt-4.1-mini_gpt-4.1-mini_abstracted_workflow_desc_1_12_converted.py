async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Express the complex number z in polar form using the given modulus |z|=4, "
        "i.e., write z = 4e^{iθ} with θ in [0, 2π). Clearly state the domain of θ and emphasize that this parametrization restricts z to the circle of radius 4 in the complex plane. "
        "Avoid attempting any further substitution or simplification at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, expressing z in polar form, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Rewrite the given expression (75 + 117i)z + (96 + 144i)/z in terms of θ by substituting z = 4e^{iθ} and simplifying each term separately. "
        "Carefully handle division by z and express all terms as functions of e^{iθ}. Avoid combining terms or extracting real parts in this step."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, rewriting expression in terms of θ, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and correct rewritten expression in terms of θ without combining terms or extracting real parts."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent answer for rewriting expression." + final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3a = (
        "Sub-task 3a: Derive the formula for the real part of the expression obtained in subtask_2 as a function of θ. "
        "Explicitly separate the real and imaginary components of each term, and write the real part as a sum of cosine and sine terms with coefficients. "
        "Carefully track signs and magnitudes of coefficients, avoiding any algebraic shortcuts that might introduce errors. Provide detailed step-by-step algebraic derivation."
    )
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_3a = []
    possible_thinkings_3a = []
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, deriving real part expression, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a)
        possible_thinkings_3a.append(thinking3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3a = "Given all the above thinking and answers, find the most consistent and correct real part expression as sum of cosine and sine terms with coefficients."
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo] + possible_thinkings_3a, "Sub-task 3a: Synthesize and choose the most consistent real part expression." + final_instr_3a, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_sc_instruction_3b = (
        "Sub-task 3b: Independently verify the coefficients of the cosine and sine terms in the real part expression derived in subtask_3a. "
        "Perform step-by-step algebraic checks, including sign verification and magnitude confirmation, to ensure no errors in coefficient extraction. "
        "Document each verification step explicitly to facilitate later cross-checking."
    )
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_3b = []
    possible_thinkings_3b = []
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", thinking3a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3a], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, verifying coefficients, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b)
        possible_thinkings_3b.append(thinking3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3b = "Given all the above thinking and answers, find the most consistent and correct verification of coefficients."
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo] + possible_thinkings_3b, "Sub-task 3b: Synthesize and choose the most consistent verification of coefficients." + final_instr_3b, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Combine and simplify the trigonometric terms in the real part expression into a single trigonometric function of the form R cos(θ + φ) or an equivalent sum of cosines and sines with consolidated coefficients. "
        "Use trigonometric identities carefully and verify each algebraic manipulation step explicitly. Avoid skipping intermediate steps to prevent sign or coefficient errors."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3a.content, thinking3b.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3a, thinking3b], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, combining trigonometric terms, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Given all the above thinking and answers, find the most consistent and correct combined trigonometric form with amplitude R and phase φ."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent combined trigonometric form." + final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Independently re-derive and cross-validate the combined coefficients obtained in subtask_4 by performing an algebraic verification of the amplitude R and phase φ. "
        "Confirm that the combination is consistent with the original coefficients from subtask_3b. Include numeric sanity checks by substituting specific θ values (e.g., where cos(θ + φ) = 1) to verify the real part matches expected values. Document any discrepancies and flag uncertainties explicitly. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4] + all_thinking5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, algebraic verification and numeric sanity checks, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Given all the above thinking and answers, reason over them carefully and provide a final verified algebraic and numeric validation of the combined coefficients and amplitude R and phase φ."
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1], "Sub-task 5: Final algebraic and numeric verification." + final_instr_5, is_sub_task=True)
    agents.append(f"Final Decision agent, final algebraic and numeric verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    reflexion_instruction_6 = (
        "Sub-task 6: Determine the maximum value of the real part expression over θ in [0, 2π) using the simplified trigonometric form from subtask_4. "
        "Provide a detailed explanation of the maximization process, including the identification of θ that achieves the maximum. "
        "Cross-check all previous subtasks' outputs (3a, 3b, 4, 5) to ensure consistency. "
        "If any inconsistency or error is detected, trigger a feedback loop to request re-execution or correction of relevant earlier subtasks. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking3a, thinking3b, thinking4, thinking5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": reflexion_instruction_6,
        "context": ["user query", thinking3a.content, thinking3b.content, thinking4.content, thinking5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, determining maximum and cross-checking consistency, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6], "Please review and provide detailed term-by-term verification and critique. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining maximum determination, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    reflexion_instruction_7 = (
        "Sub-task 7: Verify the maximum real part value found in subtask_6 by cross-checking it against all previous subtasks' outputs, including the original expression, coefficient derivations, and combined trigonometric form. "
        "Perform numeric sanity checks at critical points and confirm consistency of all algebraic steps. "
        "If any inconsistency or error is detected, trigger a feedback loop to request re-execution or correction of relevant earlier subtasks (especially subtasks 3a, 3b, 4, and 5). "
        "Document the verification process thoroughly. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking3a, thinking3b, thinking4, thinking5, thinking6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": reflexion_instruction_7,
        "context": ["user query", thinking3a.content, thinking3b.content, thinking4.content, thinking5.content, thinking6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflexion_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying maximum value and consistency, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7], "Please review and provide detailed verification and critique. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflexion_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
