async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Derive coordinate or vector representations for rectangles ABCD and EFGH using the given side lengths and rectangle properties. "
        "Assign coordinates to points A, B, C, D, E, F, G, H consistent with the rectangles' definitions and given side lengths (AB=107, BC=16, EF=184, FG=17). "
        "Establish a coordinate system or reference frame to facilitate further geometric analysis. Avoid arbitrary orientations without justification; consider standard orientation unless contradicted by constraints."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving coordinate representations, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Formulate the geometric constraints from the problem: (1) collinearity of points D, E, C, F, and (2) cyclic condition of points A, D, H, G. "
        "Express these constraints mathematically using the coordinate representations from subtask 1. "
        "Write equations for collinearity (e.g., slope equality or vector linear dependence) and the cyclic quadrilateral condition (e.g., equal opposite angles or power of point). "
        "Avoid making assumptions about point order on the circle or line without verification."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, formulating geometric constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent geometric constraints formulation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Identify and verify the positions of points E, F, G, and H relative to rectangle EFGH and the line containing D, E, C, F, ensuring consistency with the collinearity and cyclic conditions. "
        "Confirm that the rectangles' properties (right angles, equal opposite sides) hold under these constraints. Enumerate possible configurations and select those compatible with all given conditions. "
        "Avoid ignoring any constraints or accepting contradictory configurations."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, verifying point positions and configurations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Select and verify compatible configurations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Verify the cyclic quadrilateral condition for points A, D, H, G by checking the equality of opposite angles or using the power of a point theorem. "
        "Confirm that the chosen configuration from subtask 1 satisfies this condition. This verification ensures the geometric consistency of the problem setup before proceeding to numeric computations."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, verifying cyclic quadrilateral condition, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Confirm cyclic quadrilateral condition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_2_1 = (
        "Sub-task 1: Decompose the segment CE into components expressible in terms of known lengths and variables derived from the coordinate setup and constraints. "
        "Use the collinearity of D, E, C, F to express CE as a sum or difference of segments along the line. Simplify these expressions to minimal form, possibly introducing ratios or parameters to relate unknown lengths to known side lengths. "
        "Avoid introducing unnecessary complexity or unverified assumptions."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_rounds_2_1)]
    all_answer_2_1 = [[] for _ in range(N_rounds_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, decomposing CE, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Synthesize decomposition of CE.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    debate_instruction_2_2 = (
        "Sub-task 2: Simplify and compute intermediate lengths or ratios obtained in subtask 1 using the rectangle properties and cyclic quadrilateral constraints. "
        "Apply Ptolemy's theorem, similarity, or trigonometric relations to relate the segments. Ensure all simplifications are mathematically valid and consistent with previous results."
    )
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_2_2 = self.max_round
    all_thinking_2_2 = [[] for _ in range(N_rounds_2_2)]
    all_answer_2_2 = [[] for _ in range(N_rounds_2_2)]
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_2_2):
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_1, answer_2_1], debate_instruction_2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1, answer_2_1] + all_thinking_2_2[r-1] + all_answer_2_2[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying and computing intermediate lengths, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_2[r].append(thinking_i)
            all_answer_2_2[r].append(answer_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1] + all_answer_2_2[-1], "Sub-task 2: Synthesize simplifications and computations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate the simplified components and intermediate values from stage 2 to compute the final length of segment CE. "
        "Perform any necessary arithmetic operations to combine these values into a single numeric result. "
        "Verify the final answer for consistency with the problem's geometric constraints and given data."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_1_2.content, answer_1_2.content, thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_0_1, answer_0_1, thinking_1_2, answer_1_2, thinking_2_2, answer_2_2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating and computing final CE, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_sc_instruction_3_2 = (
        "Sub-task 2: Provide a final verification of the computed length CE by cross-checking with alternative geometric methods or by substituting back into the original constraints (collinearity and cyclic conditions). "
        "Confirm that the solution is unique and consistent. Present the final answer clearly."
    )
    N_sc_3_2 = self.max_sc
    cot_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_2)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc_3_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3_2):
        thinking_i, answer_i = await cot_agents_3_2[i]([taskInfo, thinking_3_1, answer_3_1], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_2[i].id}, verifying final CE, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_2.append(answer_i)
        possible_thinkings_3_2.append(thinking_i)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + possible_answers_3_2 + possible_thinkings_3_2, "Sub-task 2: Final verification and confirmation of CE.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
