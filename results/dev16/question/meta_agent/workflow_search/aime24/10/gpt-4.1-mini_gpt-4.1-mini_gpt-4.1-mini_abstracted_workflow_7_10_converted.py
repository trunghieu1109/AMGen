async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Coordinate Setup and Geometric Condition Formulation

    # Sub-task 1: Establish general coordinate system with parametric collinearity line (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Assign coordinates to rectangle ABCD with A at origin and AB along x-axis. "
        "Represent the collinearity line through points D, E, C, F parametrically with slope m and intercept b. "
        "Express points E and F as parameters on this line. Avoid fixing the line orientation or EF alignment. "
        "Provide general coordinate expressions for all points involved, maintaining full generality."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, establishing coordinate system and parametric line, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Formulate collinearity condition algebraically (CoT)
    cot_instruction_2 = (
        "Sub-task 2: Using the parametric line from Sub-task 1, formulate the algebraic collinearity condition for points D, E, C, and F. "
        "Express E and F coordinates as parameters on the line. Ensure the condition is explicit and general, without numeric assumptions."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formulating collinearity condition, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Formulate cyclic quadrilateral condition algebraically (CoT)
    cot_instruction_3 = (
        "Sub-task 3: Using coordinates from Sub-task 1, derive the algebraic condition for points A, D, H, and G to lie on the same circle. "
        "Express this cyclic condition without numeric assumptions, linking the two rectangles."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formulating cyclic quadrilateral condition, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Symbolic Relations and Expressions

    # Sub-task 4: Enumerate and relate all geometric constraints (SC_CoT)
    cot_sc_instruction_4 = (
        "Sub-task 4: Based on outputs from Sub-tasks 2 and 3, enumerate all geometric constraints including side lengths, perpendicularity, collinearity, and cyclic conditions. "
        "Symbolically relate unknown coordinates and parameters without numeric assumptions."
    )
    N_sc = self.max_sc
    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking2.content, thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_sc_agents_4[i]([taskInfo, thinking2, thinking3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, enumerating geometric constraints, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent geometric constraints relations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Express unknowns and CE in terms of given lengths and parameters (SC_CoT)
    cot_sc_instruction_5 = (
        "Sub-task 5: Using relations from Sub-task 4, express all unknown coordinates and the length CE symbolically in terms of given side lengths and parameters. "
        "Prepare explicit algebraic expressions suitable for numeric evaluation, maintaining generality."
    )
    cot_sc_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_sc_agents_5[i]([taskInfo, thinking4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_5[i].id}, expressing unknowns and CE, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent expressions for unknowns and CE.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 3: Solve, Verify, and Reflect

    # Sub-task 6: Solve algebraic system rigorously (CoT)
    cot_instruction_6 = (
        "Sub-task 6: Solve the algebraic system from Sub-task 5 exactly, retaining ±√ terms until final numeric evaluation. "
        "Compute all candidate roots for parameters and length CE. Avoid truncation or premature discarding of roots."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, solving algebraic system rigorously, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Verify candidate solutions against all constraints (SC_CoT)
    cot_sc_instruction_7 = (
        "Sub-task 7: Verify each candidate solution from Sub-task 6 by substituting back into all original geometric constraints, including rectangle properties, collinearity, and cyclic conditions. "
        "Check consistency within numerical tolerance and reject invalid solutions. Flag if no valid solution is found."
    )
    cot_sc_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_7 = []
    possible_thinkings_7 = []
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking7, answer7 = await cot_sc_agents_7[i]([taskInfo, thinking6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_7[i].id}, verifying candidate solutions, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 7: Synthesize and select valid solutions after verification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Synthetic geometry checks on verified solutions (SC_CoT)
    cot_sc_instruction_8 = (
        "Sub-task 8: Perform synthetic geometry checks on verified solutions, such as confirming opposite angles of cyclic quadrilateral sum to 180°, and rectangle side orientations and lengths are consistent. "
        "Use these checks to cross-validate coordinate geometry results and confirm length CE."
    )
    cot_sc_agents_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_8 = []
    possible_thinkings_8 = []
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking8, answer8 = await cot_sc_agents_8[i]([taskInfo, thinking7], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_8[i].id}, performing synthetic geometry checks, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8)
        possible_thinkings_8.append(thinking8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + possible_thinkings_8, "Sub-task 8: Synthesize and confirm geometric validity of solutions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Reflect on entire solution and propose refinements (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_9 = (
        "Sub-task 9: Reflect on the entire solution process and results, analyzing geometric feasibility and correctness of computed length CE. "
        "If contradictions or inconsistencies are detected, propose alternative orientations or parameter values and outline iterative refinement plans. "
        + reflect_inst
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_9 = [taskInfo, thinking1, thinking2, thinking3, thinking4, thinking5, thinking6, thinking7, thinking8]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query"] + cot_inputs_9,
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, reflecting on solution, thinking: {thinking9.content}; answer: {answer9.content}")
    critic_inst_9 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_reflect):
        feedback9, correct9 = await critic_agent_9([taskInfo, thinking9], critic_inst_9, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs_9.extend([thinking9, feedback9])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining solution, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
