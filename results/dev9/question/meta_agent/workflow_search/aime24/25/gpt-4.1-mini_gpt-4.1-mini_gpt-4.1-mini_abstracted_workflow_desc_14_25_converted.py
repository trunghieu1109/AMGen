async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Vector Parametrization and Intersection Points
    cot_instruction_1 = (
        "Sub-task 1: Establish a concrete coordinate system and parametrization for the hexagon. "
        "Fix vertex A at (0,0) and vertex B at (s,0), where s is the unknown side length. "
        "Define angles θ = angle ABC and φ = angle BCD to determine directions of sides CD and EF. "
        "Express all six sides AB, BC, CD, DE, EF, FA as vectors of length s with directions consistent with the parallelism conditions: AB ∥ DE, BC ∥ EF, CD ∥ FA. "
        "Write explicit vector coordinates for each side in terms of s, θ, and φ."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, parametrizing hexagon, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_instruction_2 = (
        "Sub-task 2: Using the vector parametrization from Sub-task 1, write explicit parametric equations for the lines containing sides AB, CD, and EF. "
        "Derive formulas for the intersection points of these extended lines pairwise (AB & CD, CD & EF, EF & AB) as functions of s, θ, and φ. "
        "Document algebraic expressions for these intersection points to enable exact distance calculations. "
        "Respect convexity, orientation, and parallelism constraints."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, deriving intersection points, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    cot_instruction_3 = (
        "Sub-task 3: Formulate closed-form expressions for the distances between the three intersection points found in Sub-task 2. "
        "Express these distances as functions of s, θ, and φ. "
        "Identify geometric constraints (closure, angle sums, parallelism) to reduce variables. "
        "Avoid numerical substitution until symbolic expressions are fully derived."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formulating distances, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # Stage 2: Equation Setup and Solving
    cot_sc_instruction_1 = (
        "Sub-task 1: Set up the system of equations by equating the three distances from Stage 1 Subtask 3 to the given triangle side lengths 200, 240, and 300. "
        "Incorporate geometric constraints such as hexagon closure, convexity, and parallelism to reduce variables and ensure consistency. "
        "Formulate the system explicitly in terms of s, θ, and φ."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_4 = []
    possible_thinkings_4 = []
    for i in range(self.max_sc):
        thinking4, answer4 = await cot_sc_agents_1[i]([taskInfo, thinking3, answer3], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, setting and solving equations, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, "Sub-task 2: Synthesize and choose the most consistent solution for s, θ, and φ.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    # Stage 3: Numeric Verification and Reflection
    debate_instruction_1 = (
        "Sub-task 1: Perform detailed numeric verification of the solution obtained in Stage 2 Subtask 2. "
        "Substitute computed values of s, θ, and φ back into vector and line equations to confirm distances between intersection points match 200, 240, and 300 within tolerance. "
        "Verify hexagon is convex, equilateral, and has stated parallelism. "
        "Discuss any discrepancies or rounding errors and provide final numeric value of s. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc5 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_1,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_1, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, numeric verification, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_3[r].append(thinking5)
            all_answer_3[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 3: Final numeric verification and answer synthesis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    reflect_instruction = (
        "Sub-task 2: Reflect on the entire solution process to confirm all problem constraints are respected and the final answer is robust and consistent. "
        "Address possible ambiguities such as vertex labeling or alternative configurations and explain why the chosen parameterization and solution are valid. "
        "Avoid introducing new assumptions or changing problem conditions. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_reflect = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_reflect = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": reflect_instruction,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4.content, answer4.content, thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_reflect(cot_inputs_reflect, reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, reflecting on solution, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_reflect([taskInfo, thinking6, answer6], "Please review and provide limitations of the provided solution. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_reflect.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_reflect.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_reflect(cot_inputs_reflect, reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, refining solution, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
