async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent the family F of unit-length segments PQ with P=(x,0), Q=(0,y), x,y>0, and x^2 + y^2 = 1. "
        "Express the parametric form of these segments and characterize the set of points covered by F. "
        "Avoid attempting to solve for the unique point at this stage; focus solely on the formal representation and geometric interpretation."
    )
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, representing family F, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Parametrize the segment AB with endpoints A=(1/2,0) and B=(0,sqrt(3)/2). "
        "Express any point C on AB (excluding endpoints) in terms of a parameter t in (0,1). "
        "Avoid solving for t yet; focus on the parametric form and geometric properties of AB."
    )
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parametrizing AB, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_0_3 = (
        "Sub-task 3: Formulate the condition for a point C on AB to lie on some segment PQ in F other than AB. "
        "Express C as a convex combination of P and Q with P and Q on the axes and x^2 + y^2 = 1. "
        "Avoid attempting to find the unique C yet; focus on deriving the necessary algebraic conditions."
    )
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1, thinking_0_2],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent([taskInfo, thinking_0_1, thinking_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating condition for C on AB, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Stage 0 Subtask 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    N_sc = self.max_sc
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]

    cot_sc_instruction_1 = (
        "Stage 1 Sub-task 1: Analyze the algebraic conditions derived in Stage 0 to characterize the set of points on AB covered by segments in F other than AB. "
        "Identify the uniqueness condition for the point C that is not covered by any other segment. "
        "Avoid computing explicit numeric values; focus on logical and algebraic characterization."
    )
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0_3],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_sc[i]([taskInfo, thinking_0_3], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing coverage on AB, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Stage 1 Sub-task 1: Synthesize and choose the most consistent and correct characterization of coverage on AB.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Stage 1 Sub-task 2: Derive an explicit equation or system of equations to solve for the parameter t corresponding to the unique point C on AB that is not covered by any other segment in F. "
        "Avoid solving the system numerically at this stage; focus on obtaining a solvable form."
    )
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1_1],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_sc[i]([taskInfo, thinking_1_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, deriving equation for t, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Stage 1 Sub-task 2: Synthesize and choose the most consistent and correct solvable equation for t.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Stage 2 Sub-task 1: Solve the equation(s) obtained in Stage 1 to find the exact coordinates of the unique point C on AB. "
        "Compute OC^2 as a reduced fraction p/q with positive coprime integers p and q. "
        "Avoid approximations; ensure exact simplification."
    )
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent([taskInfo, thinking_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, solving for C and OC^2, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = (
        "Stage 2 Sub-task 2: Calculate the sum p+q from the reduced fraction p/q obtained for OC^2. "
        "This is the final answer to the problem. Avoid re-deriving p and q; use the simplified fraction from the previous subtask. "
        + reflect_inst
    )
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    thinking_2_2, answer_2_2 = await cot_agent_reflect([taskInfo, thinking_2_1, answer_2_1], cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, calculating p+q, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")

    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking_2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        thinking_2_2, answer_2_2 = await cot_agent_reflect([taskInfo, thinking_2_1, answer_2_1, thinking_2_2, feedback], cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, refining p+q calculation, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")

    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
