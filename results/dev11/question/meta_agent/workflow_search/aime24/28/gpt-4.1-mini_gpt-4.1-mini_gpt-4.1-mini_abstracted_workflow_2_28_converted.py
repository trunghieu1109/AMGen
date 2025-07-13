async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Define geometric setup and clarify tangency configurations
    cot_instruction_1 = (
        "Sub-task 1: Precisely define the geometric setup of the problem. "
        "Confirm torus major radius R=6, minor radius r=3, sphere radius 11. "
        "Establish coordinate system and parametric equations for torus and sphere. "
        "Explicitly clarify the phrase 'when T rests on the outside of S' appearing twice by defining two distinct tangency configurations: "
        "(1) inner tangency where the torus contacts the sphere on the inside of its tube, "
        "and (2) outer tangency where the torus contacts the sphere on the outside of its tube. "
        "Avoid assumptions about orientation without justification. This sets the foundation for subsequent computations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining geometric setup and clarifying tangency configurations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)

    # Stage 1 Sub-task 2: Derive and solve tangency conditions for inner and outer tangency
    cot_sc_instruction_2 = (
        "Sub-task 2: For each tangency configuration (inner and outer), establish relative positions of torus and sphere centers consistent with external tangency. "
        "Define vertical offset d along torus axis. Derive geometric conditions relating d, angle v on torus cross-section, and sphere radius. "
        "Solve symbolically and numerically for d, v_i (inner), and v_o (outer). Verify cosine values are in [-1,1]. "
        "Reject invalid assumptions. Treat inner and outer cases separately."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, solving tangency conditions for inner and outer cases, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent and correct solutions for d, v_i, v_o.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)

    # Stage 2 Sub-task 1: Compute radii r_i and r_o of tangent circles
    cot_instruction_3 = (
        "Sub-task 3: Using solved values of d, v_i, and v_o, compute radii r_i = R + r cos v_i and r_o = R + r cos v_o of tangent circles on torus surface. "
        "Maintain symbolic expressions for exactness. Verify these radii correspond to valid geometric circles and tangency conditions hold."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2.content, answer2.content], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computing radii r_i and r_o, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)

    # Stage 2 Sub-task 2: Derive exact expression for difference r_i - r_o and simplify
    cot_instruction_4 = (
        "Sub-task 4: Derive exact expression for difference r_i - r_o in terms of known parameters and cos v_i, cos v_o. "
        "Perform algebraic simplification to express difference as reduced fraction m/n with m,n relatively prime positive integers. "
        "Justify each step carefully, avoid rounding until fraction fully simplified. Prepare fraction for final verification."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3.content, answer3.content], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, deriving and simplifying difference r_i - r_o, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)

    # Stage 3 Sub-task 1: Final verification and compute m + n
    debate_instruction_5 = (
        "Sub-task 5: Perform final verification by numerically substituting all computed values (d, v_i, v_o, r_i, r_o) into original tangency conditions. "
        "Confirm external tangency along circles of computed radii within tolerance. Verify fraction m/n is lowest terms and consistent. "
        "After verification, compute and report final answer m + n with clear summary of key intermediate values and final result."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4.content, answer4.content], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4.content, answer4.content] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying and finalizing answer, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Final verification and answer synthesis. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, computing final verified answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs