async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Vector Representation and Verification

    # Sub-task 1: Represent the hexagon side vectors as three pairs of opposite sides (u, -u), (v, -v), (w, -w), clarify equilateral and closure conditions without assuming u + v + w = 0
    cot_instruction_1 = (
        "Sub-task 1: Formally represent the convex equilateral hexagon ABCDEF as six side vectors arranged in three pairs of opposite sides: (u, -u), (v, -v), and (w, -w). "
        "State that all sides have equal length s, opposite sides are parallel and equal in magnitude but opposite in direction, and the sum of all six side vectors is zero (closure condition). "
        "Explicitly clarify that u + v + w is not necessarily zero. Define notation and assumptions clearly to avoid previous errors."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, representing hexagon vectors, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Verify and challenge the assumption u + v + w = 0 explicitly
    cot_sc_instruction_2 = (
        "Sub-task 2: Using the vector representation from Sub-task 1, explicitly verify and challenge the assumption that u + v + w = 0. "
        "Analyze the hexagon's closure condition and arrangement of side vectors to demonstrate why this assumption is false. "
        "Clarify the correct vector sum conditions and document implications for further reasoning."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, verifying vector sum assumption, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent answer for vector sum verification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Parametrize lines AB, CD, EF in terms of s, u, v, w
    cot_sc_instruction_3 = (
        "Sub-task 3: Express the lines containing sides AB, CD, and EF parametrically in terms of vectors u, v, w and hexagon side length s. "
        "Represent line AB as passing through point A with direction u, line CD through point C with direction v, and line EF through point E with direction w (or appropriate directions based on labeling). "
        "Avoid assuming any special relations beyond those established. Prepare parametric equations for intersection computations."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1], cot_sc_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, parametrizing lines, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compute intersection points P, Q, R explicitly
    cot_sc_instruction_4 = (
        "Sub-task 4: Compute explicitly the intersection points P = AB ∩ CD, Q = CD ∩ EF, and R = EF ∩ AB using parametric line equations from Sub-task 3. "
        "Express these points in terms of s, u, v, w, and scalar parameters. Derive intersection formulas rigorously without shortcuts."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3], cot_sc_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, computing intersections, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Derive expressions for triangle side lengths |PQ|, |QR|, |RP| in terms of s and angles
    cot_sc_instruction_5 = (
        "Sub-task 5: Derive expressions for the side lengths of triangle PQR (|PQ|, |QR|, |RP|) in terms of hexagon side length s and angles between u, v, w. "
        "Use intersection point formulas from Sub-task 4 and vector norms. Base derivations on first principles and explicit vector calculations."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4], cot_sc_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, deriving triangle side lengths, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Algebraic System Formation and Solution

    # Sub-task 6: Formulate algebraic system equating triangle side lengths to 200, 240, 300
    cot_sc_reflect_instruction_6 = (
        "Sub-task 6: Formulate the system of algebraic equations by equating derived triangle side lengths from Sub-task 5 to given lengths 200, 240, and 300. "
        "Clearly state unknowns (s and angles between u, v, w) and constraints from hexagon properties. Avoid oversimplifications. Prepare system for symbolic or numeric solution."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_reflect_instruction_6,
        "context": ["user query", thinking5.content, thinking2.content],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    for i in range(N_sc):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5, thinking2], cot_sc_reflect_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, formulating algebraic system, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_thinkings_6, "Sub-task 6: Synthesize algebraic system formulation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Analyze and simplify algebraic system to isolate s
    cot_sc_reflect_instruction_7 = (
        "Sub-task 7: Analyze and simplify algebraic system from Sub-task 6 to isolate hexagon side length s. "
        "Identify parameterizations or substitutions to reduce complexity. Avoid unjustified identities. Document assumptions and verify validity."
    )
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_7 = []
    possible_thinkings_7 = []
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_reflect_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    for i in range(N_sc):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6], cot_sc_reflect_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, simplifying algebraic system, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 7: Synthesize simplification and isolation of s.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Solve equations to compute exact hexagon side length s
    cot_reflect_instruction_8 = (
        "Sub-task 8: Solve simplified algebraic or trigonometric equations to compute exact hexagon side length s. "
        "Verify solution consistency with convexity, equilateral, and parallelism conditions. Use numeric or symbolic methods. Avoid accepting solutions without geometric validation."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, solving for s, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback8, correct8 = await critic_agent_8([taskInfo, thinking8], "Please review and provide limitations of solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, feedback: {feedback8.content}; correct: {correct8.content}")
        if correct8.content == "True":
            break
        cot_inputs_8.extend([thinking8, feedback8])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining solution, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    # Stage 3: Interpretation and Finalization

    # Sub-task 9: Interpret computed side length s, confirm all conditions, provide final answer and reflexion
    debate_instr_9 = (
        "Sub-task 9: Interpret computed hexagon side length s in context of original problem. "
        "Confirm it satisfies all given conditions including triangle side lengths from extended lines. "
        "Provide final concise answer and discuss geometric insights. Include reflexion on solution process, verification of assumptions, and avoidance of previous errors. "
        "Given solutions from other agents, consider their opinions as advice and provide updated answer."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instr_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking8], debate_instr_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking8] + all_thinking9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instr_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1], "Sub-task 9: Provide final answer and reflexion.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
