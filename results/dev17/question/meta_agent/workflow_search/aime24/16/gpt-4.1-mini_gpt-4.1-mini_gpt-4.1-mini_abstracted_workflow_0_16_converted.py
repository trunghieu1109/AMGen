async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Problem Setup and Formula Derivations
    # Sub-task 1: Define all key elements and state perpendicularity condition (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Formally define triangle ABC with vertices A, B, C; points O (circumcenter) and I (incenter); "
        "given circumradius R=13 and inradius r=6. Clearly state the geometric constraint IA perpendicular to OI "
        "without assuming point positions or angle measures."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining problem elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Summarize all relevant geometric and trigonometric formulas (CoT)
    cot_instruction_2 = (
        "Sub-task 2: Summarize all known geometric and trigonometric formulas relating circumradius R, inradius r, side lengths, "
        "and angles of triangle ABC relevant to expressing AB·AC or relating given data to sides and angles. Include law of sines, law of cosines, "
        "formulas for incenter coordinates, and standard identities. Avoid unverified formulas."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, summarizing formulas, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Derive full closed-form identity for sin B · sin C (SC_CoT)
    cot_sc_instruction_3 = (
        "Sub-task 3: Derive and prove the full closed-form trigonometric identity sin B · sin C = (cos(B - C) + cos A) / 2, "
        "without assuming B = C or dropping terms. This identity is critical for expressing AB·AC in terms of cos A and cos(B - C)."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, deriving sinB·sinC identity, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent and correct derivation for sinB·sinC identity.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_3.id}, synthesizing sinB·sinC identity, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Derive explicit formula for AB·AC in terms of R, cos A, and cos(B - C) (SC_CoT)
    cot_sc_instruction_4 = (
        "Sub-task 4: Using the identity from Sub-task 3 and the law of sines, derive an explicit formula for AB·AC in terms of circumradius R, cos A, and cos(B - C). "
        "Express AB·AC = 2 R^2 [cos A + cos(B - C)] or equivalent, carefully verifying each algebraic step without unjustified simplifications."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, deriving AB·AC formula, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent and correct formula for AB·AC.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_4.id}, synthesizing AB·AC formula, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Derive explicit coordinate expressions for O, A, and I (SC_CoT)
    cot_sc_instruction_5 = (
        "Sub-task 5: Place triangle in coordinate system with circumcenter O at origin and vertex A on circumcircle. "
        "Parametrize A accordingly. Derive explicit coordinate expressions for points O, A, and incenter I in terms of side lengths and angles, "
        "using angle bisector properties and known formulas for incenter coordinates. Avoid unproven simplifications like AI = r / sin(A/2). "
        "Produce explicit vector or coordinate forms for I and A suitable for applying perpendicularity condition."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, deriving coordinates for O, A, I, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent and correct coordinate expressions for O, A, and I.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_5.id}, synthesizing coordinates, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Translate perpendicularity IA ⟂ OI into explicit algebraic equation (CoT)
    cot_instruction_6 = (
        "Sub-task 6: Using coordinate expressions from Sub-task 5, translate the perpendicularity condition IA ⟂ OI into an explicit algebraic equation. "
        "Apply the dot product condition (IA) · (OI) = 0 to obtain a solvable equation relating cos A and cos(B - C). Derive this relation carefully without assuming angle equalities or dropping terms."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, translating perpendicularity condition, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Solve system for cos A and cos(B - C) (Reflexion)
    reflect_inst_7 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, solve the system of equations from Sub-tasks 4 and 6 to determine explicit numeric or symbolic values for cos A and cos(B - C), "
        "consistent with R=13, r=6, and the perpendicularity condition. Verify all algebraic steps and ensure solutions satisfy triangle inequalities and angle constraints. "
        "Avoid extraneous or invalid solutions."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking4, thinking6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": reflect_inst_7,
        "context": ["user query", thinking4.content, thinking6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflect_inst_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, solving system for cos A and cos(B-C), thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7], "Please review and provide limitations of the solution. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, feedback: {feedback7.content}; correctness: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflect_inst_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining solution, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Verify numeric plausibility and geometric consistency (Reflexion)
    reflect_inst_8 = (
        "Given previous attempts and feedback, carefully verify the numeric plausibility and geometric consistency of derived values for cos A, cos(B - C), "
        "and intermediate quantities. Check that these correspond to a valid triangle with given inradius and circumradius, and that the perpendicularity condition is satisfied. "
        "Critically assess reasonableness before computing AB·AC."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": reflect_inst_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, reflect_inst_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, verifying numeric plausibility, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback8, correct8 = await critic_agent_8([taskInfo, thinking8], "Please review and provide limitations of the verification. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, feedback: {feedback8.content}; correctness: {correct8.content}")
        if correct8.content == "True":
            break
        cot_inputs_8.extend([thinking8, feedback8])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, reflect_inst_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining verification, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    # Stage 2: Final Computation
    # Sub-task 9: Compute exact value of AB·AC using verified values (CoT)
    cot_instruction_9 = (
        "Sub-task 9: Using verified values of cos A and cos(B - C) from previous subtasks, compute the exact value of AB·AC using the formula derived in Sub-task 4. "
        "Present the final numeric answer clearly, confirming it satisfies all given conditions including IA ⟂ OI, R=13, and r=6."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, computing final AB·AC, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
