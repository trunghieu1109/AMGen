async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1.1: Model the convex equilateral hexagon ABCDEF with side length s, introduce angle parameters θ, φ, and record parallelism: AB ∥ DE, BC ∥ EF, CD ∥ FA. Ensure 0<θ<φ<π and φ−θ<π."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Given all the above thinking and answers, find the most consistent and correct modeling of the convex equilateral hexagon with parameters s, θ, φ."
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1.1: Synthesize and choose the most consistent answer. " + final_instr_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2 = (
        "Sub-task 2.1: Choose a convenient coordinate system: place A at the origin, AB along the x-axis, and express successive vertices B, C, D, E, F in terms of s, θ, and φ. Derive unit direction vectors for AB, BC, CD, EF." +
        debate_instr
    )
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking2 = [[] for _ in range(N_max)]
    all_answer2 = [[] for _ in range(N_max)]
    subtask_desc2 = {"subtask_id": "subtask_2_1", "instruction": debate_instruction_2, "context": ["user query", thinking1, answer1], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(inputs, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {all_thinking2[-1][0].content}; answer - {all_answer2[-1][0].content}")
    subtask_desc2['response'] = {"thinking": all_thinking2[-1][0], "answer": all_answer2[-1][0]}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction3 = "Sub-task 3.1: Compute the intersection points P, Q, R of lines AB∩CD, CD∩EF, EF∩AB symbolically in terms of s, θ, and φ. Derive expressions for distances PQ, QR, RP."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3_1",
        "instruction": cot_instruction3,
        "context": ["user query", thinking1, answer1, all_thinking2[-1][0], all_answer2[-1][0]],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent3(
        [taskInfo, thinking1, answer1, all_thinking2[-1][0], all_answer2[-1][0]],
        cot_instruction3,
        is_sub_task=True
    )
    agents.append(f"CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction4_base = "Sub-task 4.1: Solve the algebraic system for s, θ, φ (ignoring domain constraints). List all candidate solutions."
    debate_instruction_4 = debate_instruction4_base + debate_instr
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(N_max)]
    all_answer4 = [[] for _ in range(N_max)]
    subtask_desc4 = {
        "subtask_id": "subtask_4_1",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking1, answer1, all_thinking2[-1][0], all_answer2[-1][0], thinking3, answer3],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4, answer4 = await agent(
                    [taskInfo, thinking1, answer1, all_thinking2[-1][0], all_answer2[-1][0], thinking3, answer3],
                    debate_instruction_4,
                    r,
                    is_sub_task=True
                )
            else:
                inputs = [taskInfo, thinking1, answer1, all_thinking2[-1][0], all_answer2[-1][0], thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(inputs, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {all_thinking4[-1][0].content}; answer - {all_answer4[-1][0].content}")
    subtask_desc4['response'] = {"thinking": all_thinking4[-1][0], "answer": all_answer4[-1][0]}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs