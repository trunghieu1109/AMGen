async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Geometric Properties and Rhombus Conditions
    cot_instruction_1_1 = "Sub-task 1: Identify and clearly state the geometric properties of the hyperbola given by the equation x^2/20 - y^2/24 = 1, including its orientation, asymptotes, and the general form of points on the hyperbola."
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1_1, answer1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, analyzing hyperbola properties, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {
        "thinking": thinking1_1,
        "answer": answer1_1
    }
    logs.append(subtask_desc1_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_1_2 = "Sub-task 2: Define the properties of a rhombus, focusing on the conditions that all sides are equal and diagonals bisect each other at right angles. Emphasize the implications of the diagonals intersecting at the origin."
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1_2, answer1_2 = await cot_agent_1_2([taskInfo], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, analyzing rhombus properties, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {
        "thinking": thinking1_2,
        "answer": answer1_2
    }
    logs.append(subtask_desc1_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = "Sub-task 3: Establish the condition that the diagonals of the rhombus intersect at the origin and how this affects the symmetry and positioning of points A, B, C, D on the hyperbola. Consider the implications for the coordinates of these points."
    N_1_3 = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_1_3)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc1_3 = {
        "subtask_id": "subtask_1_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", "thinking of subtask 1_1", "answer of subtask 1_1", "thinking of subtask 1_2", "answer of subtask 1_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1_3):
        thinking1_3, answer1_3 = await cot_agents_1_3[i]([taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, considering symmetry and positioning, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
        possible_answers_1_3.append(answer1_3)
        possible_thinkings_1_3.append(thinking1_3)
    final_instr_1_3 = "Given all the above thinking and answers, find the most consistent and correct solutions for the positioning of points on the hyperbola."
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_3, answer1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 3: Synthesize and choose the most consistent answer for positioning" + final_instr_1_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    subtask_desc1_3['response'] = {
        "thinking": thinking1_3,
        "answer": answer1_3
    }
    logs.append(subtask_desc1_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Parameterization and Optimization
    cot_instruction_2_1 = "Sub-task 1: Express the coordinates of points A, B, C, D on the hyperbola in terms of parameters that satisfy both the hyperbola equation and the rhombus properties. Ensure that the parameterization respects the symmetry and intersection conditions."
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", "thinking of subtask 1_3", "answer of subtask 1_3"],
        "agent_collaboration": "CoT"
    }
    thinking2_1, answer2_1 = await cot_agent_2_1([taskInfo, thinking1_3, answer1_3], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, parameterizing coordinates, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {
        "thinking": thinking2_1,
        "answer": answer2_1
    }
    logs.append(subtask_desc2_1)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = "Sub-task 2: Derive the expressions for the lengths of the diagonals AC and BD using the parameterized coordinates of points A, B, C, D."
    N_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", "thinking of subtask 2_1", "answer of subtask 2_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_2_2):
        thinking2_2, answer2_2 = await cot_agents_2_2[i]([taskInfo, thinking2_1, answer2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, deriving diagonal expressions, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
        possible_answers_2_2.append(answer2_2)
        possible_thinkings_2_2.append(thinking2_2)
    final_instr_2_2 = "Given all the above thinking and answers, find the most consistent and correct expressions for the diagonal lengths."
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_2, answer2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and choose the most consistent answer for diagonal lengths" + final_instr_2_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {
        "thinking": thinking2_2,
        "answer": answer2_2
    }
    logs.append(subtask_desc2_2)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_2_3 = "Sub-task 3: Formulate the constrained optimization problem for maximizing BD^2 using the derived expressions and the rhombus side length condition (AC/2)^2 + (BD/2)^2 = s^2. Incorporate the perpendicularity of diagonals as an additional constraint."
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking2_3 = [[] for _ in range(N_max_2_3)]
    all_answer2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc2_3 = {
        "subtask_id": "subtask_2_3",
        "instruction": debate_instruction_2_3,
        "context": ["user query", "thinking of subtask 2_2", "answer of subtask 2_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking2_3, answer2_3 = await agent([taskInfo, thinking2_2, answer2_2], debate_instruction_2_3, r, is_sub_task=True)
            else:
                input_infos_2_3 = [taskInfo, thinking2_2, answer2_2] + all_thinking2_3[r-1] + all_answer2_3[r-1]
                thinking2_3, answer2_3 = await agent(input_infos_2_3, debate_instruction_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, optimizing BD^2, thinking: {thinking2_3.content}; answer: {answer2_3.content}")
            all_thinking2_3[r].append(thinking2_3)
            all_answer2_3[r].append(answer2_3)
    final_instr_2_3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer for the maximum BD^2."
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_3, answer2_3 = await final_decision_agent_2_3([taskInfo] + all_thinking2_3[-1] + all_answer2_3[-1], "Sub-task 3: Optimize BD^2" + final_instr_2_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_3 output: thinking - {thinking2_3.content}; answer - {answer2_3.content}")
    subtask_desc2_3['response'] = {
        "thinking": thinking2_3,
        "answer": answer2_3
    }
    logs.append(subtask_desc2_3)
    print("Step 6: ", sub_tasks[-1])

    # Stage 3: Solution Verification
    reflect_inst_3_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3_1 = "Sub-task 1: Solve the constrained optimization problem to find the maximum possible value of BD^2 and determine the greatest real number less than this value. Avoid assuming specific orientations unless justified by the constraints." + reflect_inst_3_1
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1 = self.max_round
    cot_inputs_3_1 = [taskInfo, thinking2_3, answer2_3]
    subtask_desc3_1 = {
        "subtask_id": "subtask_3_1",
        "instruction": cot_reflect_instruction_3_1,
        "context": ["user query", "thinking of subtask 2_3", "answer of subtask 2_3"],
        "agent_collaboration": "Reflexion"
    }
    thinking3_1, answer3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining optimization, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    critic_inst_3_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_3_1):
        feedback3_1, correct3_1 = await critic_agent_3_1([taskInfo, thinking3_1, answer3_1], "Please review and provide the limitations of provided solutions" + critic_inst_3_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, providing feedback, thinking: {feedback3_1.content}; answer: {correct3_1.content}")
        if correct3_1.content == "True":
            break
        cot_inputs_3_1.extend([thinking3_1, answer3_1, feedback3_1])
        thinking3_1, answer3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining optimization, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {
        "thinking": thinking3_1,
        "answer": answer3_1
    }
    logs.append(subtask_desc3_1)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_1, answer3_1, sub_tasks, agents)
    return final_answer, logs