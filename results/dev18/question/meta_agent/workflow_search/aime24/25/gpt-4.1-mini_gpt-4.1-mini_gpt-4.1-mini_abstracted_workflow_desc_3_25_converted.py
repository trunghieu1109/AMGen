async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = "Sub-task 0_1: Identify and clearly state all given geometric properties of the hexagon ABCDEF, including convexity, equilateral nature, and the parallelism of opposite sides. Explicitly list the pairs of opposite sides and their parallelism relations." 
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, identify geometric properties, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 0_1: Synthesize and choose the most consistent answer for identifying geometric properties.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = "Sub-task 0_2: Clarify the nature and construction of the triangle formed by the extensions of sides AB, CD, and EF. Define precisely which lines are involved and how their intersections form the triangle with side lengths 200, 240, and 300." 
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, clarify triangle construction, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 0_2: Synthesize and choose the most consistent answer for clarifying the triangle construction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = "Sub-task 0_3: Establish assumptions and constraints necessary for further analysis, such as the hexagon being planar, convex, and the triangle formed by the extended lines being non-degenerate. Avoid introducing coordinate systems or solving equations at this stage." 
    cot_sc_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "subtask_0_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_3[i].id}, establish assumptions, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 0_3: Synthesize and choose the most consistent answer for assumptions and constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0_3: ", sub_tasks[-1])

    cot_instruction_1_1 = "Sub-task 1_1: Derive a vector or coordinate representation of the hexagon ABCDEF that respects the given properties: equilateral sides and pairs of opposite sides parallel. Express the hexagon's vertices or side vectors in terms of the unknown side length and relevant parameters (e.g., angles or direction vectors)." 
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, derive vector representation, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    cot_instruction_1_2 = "Sub-task 1_2: Derive formal expressions for the lines containing sides AB, CD, and EF using the vector or coordinate representation of the hexagon. Determine the intersection points of these lines to form the triangle and express the triangle's side lengths in terms of the hexagon's side length and other parameters." 
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, derive line expressions and intersections, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = "Sub-task 1_3: Validate the derived representations by checking consistency with the given side lengths of the triangle (200, 240, 300) and the geometric constraints of the hexagon. Confirm that the triangle formed is indeed the one described in the problem." 
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "subtask_1_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_1_3[i]([taskInfo, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, validate representations, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_3.append(answer)
        possible_thinkings_1_3.append(thinking)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 1_3: Synthesize and choose the most consistent answer for validation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1_3: ", sub_tasks[-1])

    debate_instruction_2_1 = "Sub-task 2_1: Relate the triangle's side lengths (200, 240, 300) to the hexagon's side length by setting up equations based on the derived representations. Decompose these equations into simpler components or ratios that isolate the hexagon's side length. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer." 
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_2_1 = []
    all_answer_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        round_thinking = []
        round_answer = []
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_3], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3] + all_thinking_2_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, relate triangle sides to hexagon side, thinking: {thinking.content}; answer: {answer.content}")
            round_thinking.append(thinking)
            round_answer.append(answer)
        all_thinking_2_1.append(round_thinking)
        all_answer_2_1.append(round_answer)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 2_1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    debate_instruction_2_2 = "Sub-task 2_2: Simplify the equations obtained to solve explicitly for the hexagon's side length. Perform necessary algebraic manipulations and verify the solution's validity within the geometric context. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer." 
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_2_2 = []
    all_answer_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": debate_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        round_thinking = []
        round_answer = []
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_1], debate_instruction_2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1] + all_thinking_2_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplify and solve for hexagon side, thinking: {thinking.content}; answer: {answer.content}")
            round_thinking.append(thinking)
            round_answer.append(answer)
        all_thinking_2_2.append(round_thinking)
        all_answer_2_2.append(round_answer)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1], "Sub-task 2_2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2_2: ", sub_tasks[-1])

    reflect_inst_2_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_3 = "Sub-task 2_3: Compute the final numeric value of the hexagon's side length and summarize the solution, ensuring all assumptions and steps are clearly stated and justified." + reflect_inst_2_3
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_3 = [taskInfo, thinking_2_2, answer_2_2]
    subtask_desc_2_3 = {
        "subtask_id": "subtask_2_3",
        "instruction": cot_reflect_instruction_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, compute final numeric value, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent_2_3([taskInfo, thinking_2_3], "Please review and provide the limitations of provided solutions" + critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_3.extend([thinking_2_3, feedback])
        thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, refining final numeric value, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task 2_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2_3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
