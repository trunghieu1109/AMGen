async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_stage0 = (
        "Sub-task 1: Identify and verify all key elements and constraints of the problem: "
        "Confirm the hyperbola equation, properties of points A, B, C, D on the hyperbola, "
        "and rhombus conditions including equal side lengths, diagonals intersecting at the origin, "
        "and perpendicularity of diagonals. Clarify assumptions such as convexity and distinctness of points. "
        "Avoid assuming any labeling order without justification. This sets the foundation by ensuring all geometric and algebraic conditions are clearly understood and consistent."
    )
    N_sc = self.max_sc
    cot_sc_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage0 = []
    possible_thinkings_stage0 = []
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_stage0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking0, answer0 = await cot_sc_agents_stage0[i]([taskInfo], cot_sc_instruction_stage0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_stage0[i].id}, verifying problem elements, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_stage0.append(answer0)
        possible_thinkings_stage0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_answers_stage0 + possible_thinkings_stage0, "Sub-task 1: Synthesize and choose the most consistent and correct verification of problem elements and constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_stage1 = (
        "Sub-task 1: Derive parametric or coordinate representations for points A, B, C, D on the hyperbola "
        "that satisfy the midpoint and symmetry conditions implied by the diagonals intersecting at the origin. "
        "Express the diagonals as vectors and impose the perpendicularity condition. "
        "Formulate the side length equality condition algebraically. Validate these representations to ensure they correctly model the rhombus inscribed in the hyperbola with the given constraints."
    )
    cot_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_stage1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_stage1([taskInfo, thinking0, answer0], cot_instruction_stage1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage1.id}, deriving and validating coordinate representations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_stage2 = (
        "Sub-task 1: Using the derived representations, formulate explicit expressions for the squared lengths of the diagonals AC and BD, "
        "and the side length of the rhombus. Use the perpendicularity and midpoint conditions to relate these expressions. "
        "Develop an equation or system of equations linking the parameters describing the points on the hyperbola to the side length and diagonal lengths. "
        "Perform algebraic manipulation and substitution to isolate BD^2 in terms of parameters constrained by the hyperbola and rhombus properties."
    )
    N_sc2 = self.max_sc
    cot_sc_agents_stage2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc2)]
    possible_answers_stage2 = []
    possible_thinkings_stage2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_stage2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc2):
        thinking2, answer2 = await cot_sc_agents_stage2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_stage2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_stage2[i].id}, formulating and relating expressions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_stage2.append(answer2)
        possible_thinkings_stage2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_stage2 + possible_thinkings_stage2, "Sub-task 3: Synthesize and choose the most consistent and correct expressions relating BD^2 and parameters.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_stage3 = (
        "Sub-task 1: Analyze the expression for BD^2 obtained in Stage 2 to determine its supremum under the given constraints. "
        "Use optimization techniques, possibly calculus or inequality analysis, to find the greatest real number less than BD^2 for all rhombi inscribed on the hyperbola with diagonals intersecting at the origin. "
        "Verify the solution by checking boundary cases and ensuring all rhombus conditions remain satisfied. Provide the final answer along with a justification or proof of optimality."
    )
    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds = self.max_round
    all_thinking_stage3 = [[] for _ in range(N_rounds)]
    all_answer_stage3 = [[] for _ in range(N_rounds)]
    subtask_desc3 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_stage3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds):
        for i, agent in enumerate(debate_agents_stage3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_stage3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_stage3[r-1] + all_answer_stage3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_stage3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing supremum of BD^2, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage3[r].append(thinking3)
            all_answer_stage3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking_stage3[-1] + all_answer_stage3[-1], "Sub-task 4: Provide final supremum value of BD^2 and justification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
